import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import warnings
    warnings.filterwarnings("ignore")
    return go, mo, pd, px


@app.cell
def _(mo):
    mo.md(r"""
    # F1 Sponsorship Analysis — 1981–2026

    Data investigation backing the article **"Who pays for Formula 1, and what do they want for their money?"**

    Two datasets power this analysis:
    - **f1_title_sponsors.csv** — every title sponsor on the F1 grid, 1981–2026 (~601 rows)
    - **f1_full_sponsors_past6.csv** — all sponsors (all tiers) for all 10 teams, 2020–2026 (~2,079 rows)

    Industry classification uses NAICS codes (2-, 4-, and 6-digit).
    """)
    return


@app.cell
def _(pd):
    from pathlib import Path

    # Resolve dataset paths relative to this notebook file so it works
    # regardless of the working directory marimo is launched from.
    _DATASETS = Path(__file__).parent.parent / "datasets"

    title = pd.read_csv(_DATASETS / "f1_title_sponsors.csv")
    full  = pd.read_csv(_DATASETS / "f1_full_sponsors_past6.csv")

    # Basic cleanup
    title["Year"] = pd.to_numeric(title["Year"], errors="coerce")
    full["season"] = pd.to_numeric(full["season"], errors="coerce")

    # Drop rows with no sponsor data (N/A entries in title dataset)
    title_valid = title[title["title_sponsor"].notna() & (title["title_sponsor"] != "N/A")].copy()
    title_valid["sponsor_industry"] = title_valid["sponsor_industry"].fillna("Other")
    return full, title, title_valid


@app.cell
def _(mo):
    mo.md("""
    ## Dataset Overview
    """)
    return


@app.cell
def _(full, mo, title):
    mo.hstack([
        mo.stat(
            label="Title sponsor records (1981–2026)",
            value=f"{len(title):,}",
            caption=f"{title['Year'].nunique()} seasons · {title['Constructor'].nunique()} constructors"
        ),
        mo.stat(
            label="Full sponsor records (2020–2026)",
            value=f"{len(full):,}",
            caption=f"{full['season'].nunique()} seasons · {full['team_short_name'].nunique()} teams"
        ),
        mo.stat(
            label="Unique sponsor brands (2020–2026)",
            value=f"{full['sponsor_name'].nunique():,}",
            caption="across all tiers & teams"
        ),
        mo.stat(
            label="Unique title sponsors (1981–2026)",
            value=f"{title[title['title_sponsor'] != 'N/A']['title_sponsor'].nunique():,}",
            caption="distinct brand names"
        ),
    ])
    return


@app.cell
def _(full, mo, title_valid):
    tobacco_count = (title_valid["sponsor_industry"] == "Tobacco").sum()
    us_2020 = full[full["season"] == 2020]["hq_country"].str.strip().eq("USA").sum()
    us_2026 = full[full["season"] == 2026]["hq_country"].str.strip().eq("USA").sum()

    mo.hstack([
        mo.stat(
            label="Tobacco title sponsorships (1981–2026)",
            value=f"{tobacco_count}",
            caption="Marlboro, Camel, Rothmans, West, Gauloises, etc."
        ),
        mo.stat(
            label="US-HQ sponsor placements — 2020",
            value=f"{us_2020}",
            caption="across all 10 teams"
        ),
        mo.stat(
            label="US-HQ sponsor placements — 2026",
            value=f"{us_2026}",
            caption=f"+{us_2026 - us_2020} vs 2020 ({round((us_2026/us_2020-1)*100)}% growth)"
        ),
        mo.stat(
            label="2026 total placements",
            value=f"{len(full[full['season']==2026]):,}",
            caption="most commercially saturated grid in F1 history"
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## § 01 — Raw data samples

    Quick sanity-check of both datasets before analysis.
    """)
    return


@app.cell
def _(full, mo, title):
    mo.tabs({
        "Title Sponsors (first 20)": mo.plain(title.head(20)),
        "Full Sponsors (first 20)": mo.plain(full.head(20)),
        "Title Sponsors columns": mo.plain(title.dtypes.reset_index().rename(columns={"index":"column",0:"dtype"})),
        "Full Sponsors columns": mo.plain(full.dtypes.reset_index().rename(columns={"index":"column",0:"dtype"})),
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## § 02 — Title sponsor industries, 1981–2026 (Streamgraph equivalent)

    Replicates the streamgraph from the article. Each line = count of title sponsorships
    held by that industry in a given year. **Tobacco** was dominant through the 1990s;
    **Technology** has grown sharply since 2015.
    """)
    return


@app.cell
def _(px, title_valid):
    # Count title sponsorships per year using standardized NAICS 2-digit sectors.
    # naics_2digit_name is the official NAICS top-level classification — unlike
    # sponsor_industry (a personal annotation), this is fully standardized.
    stream_df = (
        title_valid
        .dropna(subset=["naics_2digit_name"])
        .groupby(["Year", "naics_2digit_name"])
        .size()
        .reset_index(name="count")
    )

    # Sort sectors by total count descending so the legend reads top-to-bottom by dominance
    naics2_order = (
        stream_df.groupby("naics_2digit_name")["count"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Colors keyed on NAICS 2-digit names as they appear in the data
    NAICS2_COLORS = {
        "Manufacturing":                                                                    "#374151",
        "Finance and Insurance":                                                            "#1d9e75",
        "Information":                                                                      "#2563eb",
        "Mining, Quarrying, and Oil and Gas Extraction":                                    "#f59e0b",
        "Transportation and Warehousing":                                                   "#0891b2",
        "Professional, Scientific, and Technical Services":                                 "#7c3aed",
        "Real Estate and Rental and Leasing":                                               "#be185d",
        "Accommodation and Food Services":                                                  "#ea580c",
        "Arts, Entertainment, and Recreation":                                              "#db2777",
        "Administrative and Support and Waste Management and Remediation Services":         "#9ca3af",
        "Educational Services":                                                             "#b45309",
        "Wholesale Trade":                                                                  "#6d28d9",
        "Retail Trade":                                                                     "#06b6d4",
    }

    fig_stream = px.area(
        stream_df,
        x="Year",
        y="count",
        color="naics_2digit_name",
        color_discrete_map=NAICS2_COLORS,
        category_orders={"naics_2digit_name": naics2_order},
        title="F1 Title Sponsor Sectors, 1981–2026  (NAICS 2-digit)",
        labels={"count": "Title Sponsorships", "Year": "Season", "naics_2digit_name": "NAICS sector"},
        template="plotly_white",
    )
    fig_stream.update_layout(
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=dict(dtick=5),
    )
    fig_stream.update_traces(line=dict(width=0.5))

    fig_stream
    return NAICS2_COLORS, naics2_order, stream_df


@app.cell
def _(mo, title_valid):
    # Tobacco = NAICS 4-digit 3122 "Tobacco Manufacturing".
    # We derive the peak from title_valid directly (not stream_df) because
    # at the 2-digit level, tobacco is buried inside "Manufacturing".
    tobacco_by_year = (
        title_valid[title_valid["naics_4digit_name"].str.contains("Tobacco", na=False)]
        .groupby("Year")
        .size()
    )
    peak_tobacco_year = int(tobacco_by_year.idxmax())
    peak_tobacco_count = int(tobacco_by_year.max())

    mo.callout(
        mo.md(
            f"**Why Manufacturing dominates the chart:** it contains every sub-sector — "
            f"beverages, auto parts, apparel, electronics, AND tobacco. Drill into the 4-digit "
            f"treemap below to see it fracture.  \n\n"
            f"*Tobacco (NAICS 3122) peaked at {peak_tobacco_count} simultaneous title sponsors "
            f"in {peak_tobacco_year}. By 2008 there were none — erased by the 2003 EU advertising directive.*"
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## § 02b — Industry treemap by year

    Replicates the interactive treemap from the article. Select a year to see the
    full title-sponsor industry mix. The top level uses 2-digit NAICS categories;
    the breakdown shows individual sponsor counts.
    """)
    return


@app.cell
def _(mo):
    year_slider = mo.ui.slider(
        start=1981,
        stop=2026,
        step=1,
        value=2026,
        label="Season",
        show_value=True,
    )
    year_slider
    return (year_slider,)


@app.cell
def _(pd, px, title_valid, year_slider):
    selected_year = year_slider.value

    year_df = title_valid[title_valid["Year"] == selected_year].copy()

    if len(year_df) == 0:
        # No data for this year — return empty figure
        fig_tree = px.treemap(
            pd.DataFrame({"industry": ["No data"], "naics_name": ["No data"], "count": [1]}),
            path=["industry"],
            values="count",
            title=f"F1 Title Sponsor Mix — {selected_year} (no data)"
        )
    else:
        # Drill: NAICS 2-digit → NAICS 4-digit → individual sponsor name.
        # All three levels use standardized NAICS codes, no personal annotations.
        tree_df = (
            year_df
            .dropna(subset=["naics_2digit_name", "naics_4digit_name"])
            .groupby(["naics_2digit_name", "naics_4digit_name", "title_sponsor"])
            .size()
            .reset_index(name="count")
        )
        fig_tree = px.treemap(
            tree_df,
            path=[px.Constant("All sectors"), "naics_2digit_name", "naics_4digit_name", "title_sponsor"],
            values="count",
            title=f"F1 Title Sponsor Mix — {selected_year}  ({len(year_df)} title sponsors, NAICS 2→4→brand)",
            color="naics_2digit_name",
            template="plotly_white",
        )

    fig_tree.update_layout(height=500, margin=dict(t=50, l=10, r=10, b=10))
    fig_tree
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## § 02c — Decade comparison: title sponsorship volumes

    The story: title sponsorship as a *structural feature* of F1 declined substantially
    from the 1980s to the 2010s. Teams replaced one large title contract with many
    smaller tiered deals.
    """)
    return


@app.cell
def _(go, title_valid):
    decade_map = {
        "1980s (1981–89)": (1981, 1989),
        "1990s (1990–99)": (1990, 1999),
        "2000s (2000–09)": (2000, 2009),
        "2010s (2010–19)": (2010, 2019),
        "2020s (2020–26)": (2020, 2026),
    }

    decade_counts = {
        label: len(title_valid[(title_valid["Year"] >= lo) & (title_valid["Year"] <= hi)])
        for label, (lo, hi) in decade_map.items()
    }

    fig_decade = go.Figure(go.Bar(
        x=list(decade_counts.keys()),
        y=list(decade_counts.values()),
        marker_color=["#c0392b", "#e67e22", "#f1c40f", "#2563eb", "#1d9e75"],
        text=list(decade_counts.values()),
        textposition="outside",
    ))
    fig_decade.update_layout(
        title="Title Sponsorships per Decade",
        yaxis_title="Count",
        xaxis_title="Decade",
        template="plotly_white",
        height=380,
        showlegend=False,
    )
    fig_decade
    return (decade_counts,)


@app.cell
def _(decade_counts, mo):
    count_80s = decade_counts["1980s (1981–89)"]
    count_10s = decade_counts["2010s (2010–19)"]
    pct_change = round((count_10s / count_80s - 1) * 100)

    mo.callout(
        mo.md(
            f"**{count_80s} title sponsorships in the 1980s → {count_10s} in the 2010s ({pct_change:+}%).**  "
            f"The decline reflects teams shifting to portfolio commercial structures: 30 smaller deals "
            f"instead of one dominant title contract."
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## § 03 — Geography of title sponsor money, 1981–2026

    Replicates the stacked area chart. Western Europe dominated for three decades.
    Asia Pacific surged after 2000. The Middle East arrived structurally after 2010.
    North America grew through tech company deals in the 2020s.
    """)
    return


@app.cell
def _(px, title_valid):
    GEO_COLORS = {
        "Western Europe": "#534AB7",
        "Asia Pacific":   "#1D9E75",
        "North America":  "#185FA5",
        "Middle East":    "#BA7517",
        "Other":          "#888880",
        "Latin America":  "#be185d",
    }

    geo_df = (
        title_valid
        .groupby(["Year", "hq_region"])
        .size()
        .reset_index(name="count")
    )

    # Calculate share per year
    geo_totals = geo_df.groupby("Year")["count"].transform("sum")
    geo_df["share"] = geo_df["count"] / geo_totals * 100

    region_order = (
        geo_df.groupby("hq_region")["count"]
        .sum().sort_values(ascending=False).index.tolist()
    )

    fig_geo = px.area(
        geo_df,
        x="Year",
        y="share",
        color="hq_region",
        color_discrete_map=GEO_COLORS,
        category_orders={"hq_region": region_order},
        title="F1 Title Sponsors by Headquarters Region, 1981–2026 (% share)",
        labels={"share": "Share of Title Sponsorships (%)", "Year": "Season", "hq_region": "Region"},
        template="plotly_white",
    )
    fig_geo.update_layout(
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=dict(dtick=5),
        yaxis=dict(ticksuffix="%"),
    )
    fig_geo.update_traces(line=dict(width=0.5))
    fig_geo
    return (geo_df,)


@app.cell
def _(geo_df, mo):
    # Find when Middle East first appears
    me_first = geo_df[geo_df["hq_region"] == "Middle East"]["Year"].min()
    me_2026 = geo_df[(geo_df["hq_region"] == "Middle East") & (geo_df["Year"] == 2026)]["count"].sum()

    mo.callout(
        mo.md(
            f"**Middle East title sponsors first appeared around {int(me_first)}.** "
            f"By 2026 they hold **{int(me_2026)} title sponsorships** — all state-owned energy companies "
            f"(Aramco, ADNOC, Orlen). Three state oil companies in three consecutive years (2021–2023) "
            f"signals a coordinated realization that F1 is the most efficient way to buy global industrial credibility."
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## § 04a — World map: sponsor HQ distribution by season (2020–2026)

    Replicates the animated world-map choropleth. Color intensity = number of
    sponsorship placements from that country per season.
    """)
    return


@app.cell
def _(mo):
    map_year_slider = mo.ui.slider(
        start=2020,
        stop=2026,
        step=1,
        value=2026,
        label="Map season",
        show_value=True,
    )
    map_year_slider
    return (map_year_slider,)


@app.cell
def _(full, map_year_slider, px):
    map_year = map_year_slider.value

    map_df = (
        full[full["season"] == map_year]
        .groupby("hq_country")
        .size()
        .reset_index(name="sponsor_count")
    )
    # Remove unknown/? entries
    map_df = map_df[~map_df["hq_country"].isin(["?", "", "N/A"])].copy()

    fig_map = px.choropleth(
        map_df,
        locations="hq_country",
        locationmode="country names",
        color="sponsor_count",
        color_continuous_scale="Blues",
        title=f"F1 Grid Sponsor HQ Countries — {map_year}",
        labels={"sponsor_count": "Sponsor placements", "hq_country": "Country"},
        template="plotly_white",
    )
    fig_map.update_layout(
        height=480,
        coloraxis_colorbar=dict(title="Placements"),
        geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
    )
    fig_map
    return


@app.cell
def _(full, px):
    # Total sponsors per season trend (line chart shown below world map in article)
    season_totals = full.groupby("season").size().reset_index(name="total_sponsors")

    fig_season_line = px.line(
        season_totals,
        x="season",
        y="total_sponsors",
        markers=True,
        title="Total Sponsorship Placements Across All Teams, 2020–2026",
        labels={"total_sponsors": "Total placements", "season": "Season"},
        template="plotly_white",
    )
    fig_season_line.update_traces(line=dict(color="#2563eb", width=2.5), marker=dict(size=8))
    fig_season_line.update_layout(height=320, xaxis=dict(dtick=1))
    fig_season_line
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## § 04b — Industry breakdown: 2020 vs 2026

    Replicates the horizontal bar chart comparing top 14 sponsor categories by
    4-digit NAICS code. Shows how the industry mix shifted from 2020 to 2026.
    """)
    return


@app.cell
def _(full, go):
    # Build 4-digit NAICS counts for 2020 and 2026
    def naics_counts(season_df, top_n=15):
        return (
            season_df
            .groupby("naics_4digit_name")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(top_n)
        )

    df_2020 = full[full["season"] == 2020]
    df_2026 = full[full["season"] == 2026]

    top14_2026 = naics_counts(df_2026, 14)
    cats_2026 = top14_2026["naics_4digit_name"].tolist()

    # Get matching counts for 2020
    counts_2020_map = df_2020.groupby("naics_4digit_name").size().to_dict()
    counts_2020 = [counts_2020_map.get(c, 0) for c in cats_2026]
    counts_2026 = top14_2026["count"].tolist()

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(
        name="2020",
        y=cats_2026,
        x=counts_2020,
        orientation="h",
        marker_color="#888880",
        opacity=0.8,
    ))
    fig_compare.add_trace(go.Bar(
        name="2026",
        y=cats_2026,
        x=counts_2026,
        orientation="h",
        marker_color="#1a1a1a",
        opacity=0.9,
    ))
    fig_compare.update_layout(
        title="Industry Breakdown: 2020 vs 2026 (Top 14 by 4-digit NAICS, 2026 count)",
        barmode="group",
        xaxis_title="Sponsor placements",
        template="plotly_white",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(autorange="reversed"),
    )
    fig_compare
    return (df_2026,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## § 04c — Top sponsor industries by high-level category (2026)

    Macro-level view using the `sponsor_industry` field — simpler than NAICS but
    tells the same story at a glance.
    """)
    return


@app.cell
def _(NAICS2_COLORS, df_2026, px):
    industry_2026 = (
        df_2026
        .dropna(subset=["naics_2digit_name"])
        .groupby("naics_2digit_name")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    fig_industry = px.bar(
        industry_2026,
        x="naics_2digit_name",
        y="count",
        color="naics_2digit_name",
        color_discrete_map=NAICS2_COLORS,
        title="2026 Grid: Sponsor Count by NAICS 2-digit Sector",
        labels={"count": "Placements", "naics_2digit_name": "NAICS sector"},
        template="plotly_white",
        text="count",
    )
    fig_industry.update_traces(textposition="outside")
    fig_industry.update_layout(
        showlegend=False,
        height=420,
        xaxis=dict(categoryorder="total descending"),
    )
    fig_industry
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## § 05 — Crypto sponsor arc, 2020–2026

    Replicates the crypto timeline grid. Each row is one crypto sponsor.
    Shows entry, active seasons, and exit. The FTX collapse (Nov 2022) is
    visible as a sharp contraction in 2023.
    """)
    return


@app.cell
def _(full, go, pd):
    crypto_df = full[full["sponsor_industry"] == "Crypto"].copy()

    # Build presence matrix
    all_years = sorted(crypto_df["season"].unique())
    all_crypto = sorted(crypto_df["sponsor_name"].unique())

    presence = pd.DataFrame(0, index=all_crypto, columns=all_years)
    for _, row in crypto_df.iterrows():
        presence.loc[row["sponsor_name"], row["season"]] = 1

    # Sort: sponsors that lasted longer toward the top
    presence = presence.loc[presence.sum(axis=1).sort_values(ascending=False).index]

    # Known collapsed sponsors
    collapsed = {"FTX", "Velas", "Bybit"}  # Bybit withdrew from F1 too

    z = []
    text = []
    colors = []
    for sponsor in presence.index:
        row_vals = presence.loc[sponsor].values
        z.append(row_vals)
        text.append([sponsor if v else "" for v in row_vals])
        colors.append(
            ["#c0392b" if v and sponsor in collapsed else ("#2563eb" if v else "#f3f4f6")
             for v in row_vals]
        )

    fig_crypto = go.Figure()

    for i, sponsor in enumerate(presence.index):
        for j, yr in enumerate(all_years):
            val = presence.loc[sponsor, yr]
            is_collapsed = sponsor in collapsed
            fill_color = (
                "#c0392b" if (val and is_collapsed) else
                "#2563eb" if val else
                "#f3f4f6"
            )
            fig_crypto.add_shape(
                type="rect",
                x0=j - 0.45, x1=j + 0.45,
                y0=i - 0.45, y1=i + 0.45,
                fillcolor=fill_color,
                line=dict(color="white", width=1),
            )
            if val:
                fig_crypto.add_annotation(
                    x=j, y=i,
                    text=sponsor if j == list(all_years).index(all_years[0]) or not presence.loc[sponsor, all_years[all_years.index(yr) - 1]] else "",
                    showarrow=False,
                    font=dict(size=9, color="white"),
                    xanchor="left",
                )

    fig_crypto.update_layout(
        title="Crypto Sponsors on the F1 Grid, 2020–2026",
        xaxis=dict(
            tickvals=list(range(len(all_years))),
            ticktext=[str(y) for y in all_years],
            title="Season",
        ),
        yaxis=dict(
            tickvals=list(range(len(presence.index))),
            ticktext=list(presence.index),
            title="",
            autorange="reversed",
        ),
        height=max(300, len(presence.index) * 36 + 80),
        template="plotly_white",
        showlegend=False,
        plot_bgcolor="#f9fafb",
        annotations=[
            dict(
                x=2.5, y=-1.2,
                text="<b>← FTX collapse (Nov 2022)</b> — Mercedes removed FTX branding mid-season",
                showarrow=False,
                font=dict(size=10, color="#c0392b"),
                xref="x", yref="y",
            )
        ]
    )
    # Add a vertical line at 2022
    fig_crypto.add_vline(
        x=list(all_years).index(2022) + 0.5 if 2022 in all_years else 2.5,
        line=dict(color="#c0392b", width=1.5, dash="dash"),
    )

    fig_crypto
    return (crypto_df,)


@app.cell
def _(crypto_df, mo):
    # Crypto stats
    crypto_by_year = crypto_df.groupby("season").size()
    peak_yr = int(crypto_by_year.idxmax())
    peak_ct = int(crypto_by_year.max())
    yr_2026_ct = int(crypto_by_year.get(2026, 0))

    mo.hstack([
        mo.stat(label="Peak crypto placements", value=f"{peak_ct}", caption=f"Season {peak_yr}"),
        mo.stat(label="Crypto placements in 2026", value=f"{yr_2026_ct}", caption=f"post-FTX survivors"),
        mo.stat(label="Unique crypto brands ever", value=f"{crypto_df['sponsor_name'].nunique()}", caption="across 2020–2026"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## § 06 — US sponsor presence, 2020–2026

    The US now hosts 3 GPs. US viewership doubled. Did US sponsor money follow?
    """)
    return


@app.cell
def _(full, go):
    us_by_year = (
        full[full["hq_country"].str.strip() == "USA"]
        .groupby("season")
        .size()
        .reset_index(name="us_count")
    )
    total_by_year = full.groupby("season").size().reset_index(name="total")
    us_share = us_by_year.merge(total_by_year, on="season")
    us_share["pct"] = us_share["us_count"] / us_share["total"] * 100

    fig_us = go.Figure()
    fig_us.add_trace(go.Bar(
        x=us_share["season"],
        y=us_share["us_count"],
        name="US-HQ placements",
        marker_color="#185FA5",
        yaxis="y",
    ))
    fig_us.add_trace(go.Scatter(
        x=us_share["season"],
        y=us_share["pct"],
        name="% of total grid",
        mode="lines+markers",
        line=dict(color="#c0392b", width=2),
        marker=dict(size=8),
        yaxis="y2",
    ))
    fig_us.update_layout(
        title="US-Headquartered Sponsor Placements, 2020–2026",
        xaxis_title="Season",
        yaxis=dict(title="Placements (count)", side="left"),
        yaxis2=dict(title="% of all grid placements", side="right", overlaying="y", ticksuffix="%"),
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(dtick=1),
    )
    fig_us
    return


@app.cell
def _(full, px):
    # US sponsors by team in 2026
    us_by_team_2026 = (
        full[(full["season"] == 2026) & (full["hq_country"].str.strip() == "USA")]
        .groupby("team_short_name")
        .size()
        .reset_index(name="us_count")
        .sort_values("us_count", ascending=False)
    )

    fig_us_team = px.bar(
        us_by_team_2026,
        x="team_short_name",
        y="us_count",
        title="US-HQ Sponsor Placements by Team — 2026",
        labels={"us_count": "US placements", "team_short_name": "Team"},
        color="us_count",
        color_continuous_scale="Blues",
        text="us_count",
        template="plotly_white",
    )
    fig_us_team.update_traces(textposition="outside")
    fig_us_team.update_layout(
        height=380,
        showlegend=False,
        coloraxis_showscale=False,
        xaxis=dict(categoryorder="total descending"),
    )
    fig_us_team
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## § 07 — Bonus findings

    Additional analysis beyond the article's published visualizations.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 7a — Sponsor tier distribution per team (2026)
    """)
    return


@app.cell
def _(df_2026, px):
    tier_order = ["Title", "Principal", "Major", "Minor"]
    tier_colors = {"Title": "#c0392b", "Principal": "#e67e22", "Major": "#2563eb", "Minor": "#9ca3af"}

    tier_team = (
        df_2026
        .groupby(["team_short_name", "sponsor_tier"])
        .size()
        .reset_index(name="count")
    )
    # Sort teams by total sponsor count
    team_order_2026 = df_2026.groupby("team_short_name").size().sort_values(ascending=False).index.tolist()

    fig_tier = px.bar(
        tier_team,
        x="team_short_name",
        y="count",
        color="sponsor_tier",
        color_discrete_map=tier_colors,
        category_orders={"team_short_name": team_order_2026, "sponsor_tier": tier_order},
        title="Sponsor Tier Breakdown by Team — 2026",
        labels={"count": "Sponsors", "team_short_name": "Team", "sponsor_tier": "Tier"},
        template="plotly_white",
        barmode="stack",
        text_auto=True,
    )
    fig_tier.update_layout(
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(categoryorder="total descending"),
    )
    fig_tier
    return


@app.cell
def _(mo):
    mo.md("""
    ### 7b — Sponsor longevity: how long brands stay on the grid (2020–2026)
    """)
    return


@app.cell
def _(full, px):
    # For each sponsor, count how many seasons they appeared
    longevity = (
        full
        .groupby("sponsor_name")["season"]
        .nunique()
        .reset_index(name="seasons_active")
        .sort_values("seasons_active", ascending=False)
    )

    longevity_dist = longevity["seasons_active"].value_counts().reset_index()
    longevity_dist.columns = ["seasons_active", "sponsor_count"]
    longevity_dist = longevity_dist.sort_values("seasons_active")

    fig_long = px.bar(
        longevity_dist,
        x="seasons_active",
        y="sponsor_count",
        title="Sponsor Longevity Distribution, 2020–2026 (how many seasons each brand stayed)",
        labels={"seasons_active": "Seasons active", "sponsor_count": "Number of sponsors"},
        color="seasons_active",
        color_continuous_scale="Blues",
        text="sponsor_count",
        template="plotly_white",
    )
    fig_long.update_traces(textposition="outside")
    fig_long.update_layout(height=360, showlegend=False, coloraxis_showscale=False)
    fig_long
    return (longevity,)


@app.cell
def _(longevity, mo):
    all_7 = longevity[longevity["seasons_active"] == 7]["sponsor_name"].tolist()
    mo.callout(
        mo.md(
            f"**{len(all_7)} sponsors appeared in all 7 seasons (2020–2026):**  \n"
            + ", ".join(sorted(all_7)[:20])
            + ("..." if len(all_7) > 20 else "")
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### 7c — Top HQ countries: sponsor concentration by nation (2026)
    """)
    return


@app.cell
def _(df_2026, px):
    country_2026 = (
        df_2026[~df_2026["hq_country"].isin(["?", "", "N/A"])]
        .groupby("hq_country")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(20)
    )

    fig_country = px.bar(
        country_2026,
        x="hq_country",
        y="count",
        title="Top 20 Sponsor HQ Countries — 2026",
        labels={"count": "Placements", "hq_country": "Country"},
        color="count",
        color_continuous_scale="Blues",
        text="count",
        template="plotly_white",
    )
    fig_country.update_traces(textposition="outside")
    fig_country.update_layout(
        height=400,
        showlegend=False,
        coloraxis_showscale=False,
        xaxis=dict(categoryorder="total descending"),
    )
    fig_country
    return


@app.cell
def _(mo):
    mo.md("""
    ### 7d — Region shift: title sponsors 2020 vs 2026 (full grid dataset)
    """)
    return


@app.cell
def _(full, go):
    reg_2020 = full[full["season"] == 2020]["hq_region"].value_counts()
    reg_2026 = full[full["season"] == 2026]["hq_region"].value_counts()
    all_regions = sorted(set(reg_2020.index) | set(reg_2026.index))

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Bar(
        name="2020",
        x=all_regions,
        y=[reg_2020.get(r, 0) for r in all_regions],
        marker_color="#888880",
    ))
    fig_reg.add_trace(go.Bar(
        name="2026",
        x=all_regions,
        y=[reg_2026.get(r, 0) for r in all_regions],
        marker_color="#185FA5",
    ))
    fig_reg.update_layout(
        title="Full Grid: Sponsor Placements by Region — 2020 vs 2026",
        barmode="group",
        template="plotly_white",
        height=380,
        yaxis_title="Placements",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_reg
    return


@app.cell
def _(mo):
    mo.md("""
    ### 7e — Tobacco & nicotine's quiet return: ZYN on Ferrari 2026
    """)
    return


@app.cell
def _(full, mo):
    tobacco_full = full[full["sponsor_industry"].isin(["Tobacco", "Other"]) & full["naics_4digit_name"].str.contains("Tobacco", na=False)]
    tobacco_by_season = tobacco_full.groupby("season")["sponsor_name"].apply(list).to_dict()

    rows = []
    for yr_t, brands in tobacco_by_season.items():
        rows.append(f"**{yr_t}:** " + ", ".join(sorted(set(brands))))

    mo.callout(
        mo.md(
            "**Tobacco and nicotine brands on the full grid (2020–2026):**\n\n"
            + "\n\n".join(rows)
            + "\n\n*Note: ZYN (nicotine pouches) and A Better Tomorrow (PMI sustainability brand) "
            "represent tobacco's continued presence under different product categories.*"
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### 7f — Which teams have the most sponsors, and how has that changed? (2020–2026)
    """)
    return


@app.cell
def _(full, px):
    team_year = (
        full
        .groupby(["season", "team_short_name"])
        .size()
        .reset_index(name="sponsor_count")
    )

    fig_team_trend = px.line(
        team_year,
        x="season",
        y="sponsor_count",
        color="team_short_name",
        markers=True,
        title="Total Sponsor Count per Team, 2020–2026",
        labels={"sponsor_count": "Sponsor placements", "season": "Season", "team_short_name": "Team"},
        template="plotly_white",
    )
    fig_team_trend.update_layout(
        height=440,
        xaxis=dict(dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_team_trend
    return


@app.cell
def _(mo):
    mo.md("""
    ### 7g — Title sponsor industry in the HISTORICAL dataset: tobacco vs non-tobacco over time
    """)
    return


@app.cell
def _(px, title_valid):
    # Binary tobacco vs. rest
    title_valid2 = title_valid.copy()
    title_valid2["is_tobacco"] = title_valid2["sponsor_industry"] == "Tobacco"
    title_valid2["category"] = title_valid2["is_tobacco"].map({True: "Tobacco", False: "Non-tobacco"})

    tob_vs_rest = (
        title_valid2
        .groupby(["Year", "category"])
        .size()
        .reset_index(name="count")
    )

    fig_tob = px.bar(
        tob_vs_rest,
        x="Year",
        y="count",
        color="category",
        color_discrete_map={"Tobacco": "#c0392b", "Non-tobacco": "#2563eb"},
        title="Tobacco vs. Non-Tobacco Title Sponsors, 1981–2026",
        labels={"count": "Title sponsorships", "Year": "Season", "category": ""},
        template="plotly_white",
        barmode="stack",
    )
    fig_tob.update_layout(
        height=380,
        xaxis=dict(dtick=5),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_tob
    return


@app.cell
def _(mo):
    mo.md("""
    ### 7h — Gambling sponsors: a rising category to watch
    """)
    return


@app.cell
def _(full, go, mo):
    gambling_df = full[full["sponsor_industry"] == "Gambling"]
    gambling_by_year = gambling_df.groupby("season")["sponsor_name"].apply(list)

    if len(gambling_df) == 0:
        mo.callout(mo.md("No gambling sponsors classified in the full dataset under 'Gambling' industry."), kind="info")
    else:
        gambling_trend = gambling_df.groupby("season").size().reset_index(name="count")

        fig_gamb = go.Figure(go.Bar(
            x=gambling_trend["season"],
            y=gambling_trend["count"],
            marker_color="#ea580c",
            text=gambling_trend["count"],
            textposition="outside",
        ))
        fig_gamb.update_layout(
            title="Gambling Sponsor Placements on the F1 Grid, 2020–2026",
            xaxis_title="Season",
            yaxis_title="Placements",
            template="plotly_white",
            height=340,
            xaxis=dict(dtick=1),
        )
        fig_gamb
    return (gambling_df,)


@app.cell
def _(gambling_df, mo):
    gambling_brands = gambling_df["sponsor_name"].unique().tolist()
    mo.callout(
        mo.md(
            f"**Gambling brands on the grid (2020–2026):** {', '.join(sorted(gambling_brands))}  \n"
            "Gambling joined crypto as the fastest-growing potentially controversial sponsor category. "
            "Unlike tobacco, it faces no sport-wide ban — yet."
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Summary of key findings

    | Finding | Data point |
    |---------|-----------|
    | Tobacco peaked | ~1999, then collapsed by 2008 following EU directive |
    | Geography shift | Western Europe → Middle East (state oil) + North America (tech) |
    | US sponsor growth | Roughly doubled 2020→2026, in line with viewership but lagging race count |
    | Crypto arc | Rose to peak ~2022, then contracted post-FTX; stabilized with survivor exchanges |
    | Title sponsor decline | Fewer per-team title deals; portfolio model now dominates |
    | Tobacco's return | ZYN (nicotine pouches) on Ferrari, A Better Tomorrow (PMI) on McLaren |
    | Gambling rising | Now a measurable category — no ban in sight |
    | McLaren leads | Most US-HQ sponsors of any team in 2026 |
    """)
    return


if __name__ == "__main__":
    app.run()
