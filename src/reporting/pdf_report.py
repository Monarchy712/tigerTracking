"""One-click survey report PDF for the forest department.

Everything on the page is pulled live from the database, so the report is
always consistent with what the dashboard is showing.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import app_config, ensure_data_dirs
from src.db.repository import Repository
from src.occupancy.live_map import (
    render_static_map_png,
    sightings_to_rows,
    territories_from_rows,
)

ACCENT = colors.HexColor("#b45309")
INK = colors.HexColor("#1f2937")
MUTED = colors.HexColor("#6b7280")
RULE = colors.HexColor("#d1d5db")


@dataclass
class ReportContext:
    """Everything the report needs, resolved once up front."""

    start: datetime | None
    end: datetime | None
    totals: dict
    rows: list[dict]
    territories: list[dict]
    alerts: list
    tigers: list
    stations: list


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "SSTitle", parent=base["Title"], fontSize=26, leading=31,
            textColor=INK, spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "SSSubtitle", parent=base["Normal"], fontSize=13, leading=18,
            textColor=MUTED, alignment=TA_CENTER, spaceAfter=4,
        ),
        "h2": ParagraphStyle(
            "SSH2", parent=base["Heading2"], fontSize=15, leading=19,
            textColor=ACCENT, spaceBefore=14, spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "SSBody", parent=base["Normal"], fontSize=9.5, leading=13, textColor=INK
        ),
        "cell": ParagraphStyle(
            "SSCell", parent=base["Normal"], fontSize=8, leading=10.5, textColor=INK
        ),
        "muted": ParagraphStyle(
            "SSMuted", parent=base["Normal"], fontSize=8.5, leading=12, textColor=MUTED
        ),
    }


def _fmt(value: datetime | None) -> str:
    return value.strftime("%d %b %Y") if isinstance(value, datetime) else "—"


def collect_context(
    repo: Repository,
    start: datetime | None = None,
    end: datetime | None = None,
) -> ReportContext:
    db_start, db_end = repo.sighting_date_bounds()
    start = start or db_start
    end = end or db_end

    tigers = repo.list_tigers()
    code_by_id = {t.id: t.tiger_code for t in tigers}
    rows = sightings_to_rows(repo.get_sightings_in_range(start, end), code_by_id)

    # Alerts are keyed by processing time, not capture time, so the capture-date
    # window does not apply to them — the whole log is ranked for the report.
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    alerts = repo.get_alerts()

    return ReportContext(
        start=start,
        end=end,
        totals=repo.count_totals(),
        rows=rows,
        territories=territories_from_rows(rows),
        alerts=sorted(
            alerts,
            key=lambda a: (severity_rank.get(a.severity, 3), -a.confidence),
        ),
        tigers=tigers,
        stations=repo.get_stations(),
    )


def _cover(ctx: ReportContext, styles: dict) -> list:
    reserve = app_config.reserve
    blanks = ctx.totals["blanks"]
    images = ctx.totals["images"]
    blank_pct = f"{blanks / images:.0%}" if images else "0%"

    story = [
        Spacer(1, 3.5 * cm),
        Paragraph("Camera Trap Survey Report", styles["title"]),
        Paragraph(reserve.name, styles["subtitle"]),
        Paragraph(
            f"Survey window: {_fmt(ctx.start)} — {_fmt(ctx.end)}", styles["subtitle"]
        ),
        Spacer(1, 1.6 * cm),
    ]

    stats = [
        ("Individuals identified", str(ctx.totals["tigers"])),
        ("Total captures", str(ctx.totals["sightings"])),
        ("Frames processed", str(images)),
        ("Blank frames removed", f"{blanks}  ({blank_pct} of intake)"),
        ("Alerts raised", str(ctx.totals["alerts"])),
        ("Matches awaiting review", str(ctx.totals["pending_reviews"])),
    ]
    table = Table(
        [[Paragraph(f"<b>{label}</b>", styles["body"]), Paragraph(value, styles["body"])]
         for label, value in stats],
        colWidths=[9 * cm, 6 * cm],
    )
    table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(table)
    story.append(Spacer(1, 2 * cm))
    story.append(
        Paragraph(
            f"Generated {datetime.now():%d %b %Y, %H:%M} · {reserve.state} · "
            "individual identification, occupancy and deviation alerting.",
            styles["muted"],
        )
    )
    story.append(PageBreak())
    return story


def _map_section(ctx: ReportContext, styles: dict, workdir: Path) -> list:
    story = [Paragraph("Reserve occupancy", styles["h2"])]
    png = render_static_map_png(
        ctx.rows, ctx.territories, workdir / "reserve_map.png", stations=ctx.stations
    )
    if png and png.exists():
        story.append(Image(str(png), width=16 * cm, height=12.8 * cm, kind="proportional"))
        story.append(Spacer(1, 0.3 * cm))
        story.append(
            Paragraph(
                "Convex-hull home range per individual, with capture locations and "
                "camera stations. Ringed markers are range centroids.",
                styles["muted"],
            )
        )
    else:
        story.append(
            Paragraph("No georeferenced captures in this window.", styles["body"])
        )
    return story


def _alerts_section(ctx: ReportContext, styles: dict, code_by_id: dict) -> list:
    limit = app_config.report.top_alerts
    story = [Paragraph(f"Priority alerts (top {limit} by severity)", styles["h2"])]

    if not ctx.alerts:
        story.append(Paragraph("No alerts raised in this window.", styles["body"]))
        return story

    header = ["Individual", "Type", "Conf.", "Severity", "Detail"]
    data = [[Paragraph(f"<b>{h}</b>", styles["cell"]) for h in header]]

    for alert in ctx.alerts[:limit]:
        evidence = ""
        if alert.evidence_json:
            try:
                parsed = json.loads(alert.evidence_json)
                evidence = " · ".join(
                    f"{k}={v}" for k, v in list(parsed.items())[:3]
                    if not isinstance(v, (list, dict))
                )
            except json.JSONDecodeError:
                evidence = ""
        detail = alert.description + (f"<br/><font size=6>{evidence}</font>" if evidence else "")
        data.append([
            Paragraph(code_by_id.get(alert.tiger_id, "—"), styles["cell"]),
            Paragraph(
                alert.alert_type.replace("anomaly:", "anomaly / ").replace("_", " "),
                styles["cell"],
            ),
            Paragraph(f"{alert.confidence:.0%}", styles["cell"]),
            Paragraph(alert.severity, styles["cell"]),
            Paragraph(detail, styles["cell"]),
        ])

    table = Table(data, colWidths=[2.0 * cm, 2.6 * cm, 1.3 * cm, 1.7 * cm, 8.4 * cm], repeatRows=1)
    table.setStyle(_table_style())
    story.append(table)
    return story


def _individuals_section(ctx: ReportContext, repo: Repository, styles: dict) -> list:
    story = [Paragraph("Individuals", styles["h2"])]
    if not ctx.tigers:
        story.append(Paragraph("No individuals enrolled yet.", styles["body"]))
        return story

    area_by_id = {t["tiger_id"]: t for t in ctx.territories}
    header = ["Code", "Captures", "Home range (km²)", "Stations", "First seen", "Last seen"]
    data = [[Paragraph(f"<b>{h}</b>", styles["cell"]) for h in header]]

    for tiger in ctx.tigers:
        sightings = repo.get_sightings_for_tiger(tiger.id)
        dates = [s.captured_at for s in sightings if s.captured_at]
        territory = area_by_id.get(tiger.id, {})
        data.append([
            Paragraph(tiger.tiger_code, styles["cell"]),
            Paragraph(str(len(sightings)), styles["cell"]),
            Paragraph(str(territory.get("area_sq_km", "—")), styles["cell"]),
            Paragraph(str(len(repo.stations_used_by_tiger(tiger.id))), styles["cell"]),
            Paragraph(_fmt(min(dates)) if dates else "—", styles["cell"]),
            Paragraph(_fmt(max(dates)) if dates else "—", styles["cell"]),
        ])

    table = Table(data, colWidths=[2.2 * cm, 2.2 * cm, 3.4 * cm, 2.2 * cm, 3 * cm, 3 * cm], repeatRows=1)
    table.setStyle(_table_style())
    story.append(table)
    return story


def _table_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("GRID", (0, 0), (-1, -1), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
    ])


def generate_survey_report(
    repo: Repository,
    output_path: Path | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Path:
    """Build the survey PDF and return its path."""
    ctx = collect_context(repo, start, end)
    styles = _styles()
    code_by_id = {t.id: t.tiger_code for t in ctx.tigers}

    if output_path is None:
        reports_dir = ensure_data_dirs()["reports"]
        output_path = reports_dir / f"survey_{date.today():%Y-%m-%d}.pdf"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        title=f"Survey Report — {app_config.reserve.name}",
        author="Tiger Tracking System",
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
    )

    with tempfile.TemporaryDirectory() as tmp:
        story = _cover(ctx, styles)
        story += _map_section(ctx, styles, Path(tmp))
        story.append(PageBreak())
        story += _alerts_section(ctx, styles, code_by_id)
        story.append(Spacer(1, 0.8 * cm))
        story += _individuals_section(ctx, repo, styles)
        doc.build(story, onLaterPages=_footer, onFirstPage=_footer)

    return output_path


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.1 * cm, app_config.reserve.name)
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()
