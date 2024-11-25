from collections import OrderedDict
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import textwrap

SCORE_MAPPING = OrderedDict([
    (4, "A"),
    (3, "B"), (3.5, "B+"),
    (2, "C"), (2.5, "C+"),
    (1, "D"), (1.5, "D+"),
])

PILLAR_MAPPING = OrderedDict([
    ('pillar1_budget_reliability', '1. Budget reliability'),
    ('pillar2_transparency', '2. Transparency'),
    ('pillar3_asset_liability', '3. Asset and liability management'),
    ('pillar4_policy_based_budget', '4. Policy-based budgeting'),
    ('pillar5_predictability_and_control', '5. Predictability and control'),
    ('pillar6_accounting_and_reporting', '6. Accounting and reporting'),
    ('pillar7_external_audit', '7. External audit'),
])


def pefa_overall_figure(df, pov_df):
    pillar_columns = [col for col in df.columns if col.startswith('pillar')]
    overall_scores = df[pillar_columns].mean(axis=1)

    overall_grades = overall_scores.map(_score_to_grade)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            name="Poverty Rate",
            x=pov_df.year,
            y=pov_df.poor215,
            mode="lines+markers",
            line=dict(color="darkred", shape="spline", dash="dot"),
            connectgaps=True,
            hovertemplate=("%{x}: %{y:.2f}%"),
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            name="PEFA Score",
            x=df.year,
            y=overall_scores,
            mode="lines+markers",
            marker_color="darkblue",
            hovertemplate=("%{x}: %{y:.2f} (%{customdata})"),
            customdata=overall_grades,
        ),
        secondary_y=False,
    )

    fig.update_xaxes(tickformat="d")
    fig.update_yaxes(
        title_text="Quality of Budget Institutions",
        secondary_y=False,
        tickvals=list(SCORE_MAPPING.keys()),
        ticktext=list(SCORE_MAPPING.values()),
        range=[0, 4.5],
    )
    fig.update_yaxes(
        title_text="Poverty Rate (%)",
        secondary_y=True,
        range=[-10, 100],
    )
    title_text = "How did the overall quality of budget institutions change over time?"
    wrapped_title = "<br>".join(textwrap.wrap(title_text, width=40))
    fig.update_layout(
        barmode="stack",
        title={
            'text': wrapped_title,
            'font': {'size': 16},
            'x': 0.5,
            'y': 0.92,
            'xanchor': 'center',
        },
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.03),
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=-0.14,
                y=-0.2,
                text="Source: PEFA & Poverty Rate: World Bank",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )
    return fig


def pefa_pillar_heatmap(df):
    heatmap_data = df.melt(
        id_vars=['year'],
        value_vars=[col for col in df.columns if col.startswith('pillar')],
        var_name='pillar',
        value_name='score'
    )

    heatmap_data['grade'] = heatmap_data['score'].map(_score_to_grade)
    heatmap_data['pillar'] = heatmap_data['pillar'].map(PILLAR_MAPPING)

    heatmap_scores = heatmap_data.pivot(index='pillar', columns='year', values='score')
    heatmap_grades = heatmap_data.pivot(index='pillar', columns='year', values='grade')

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_scores.values,
            x=heatmap_scores.columns,
            y=heatmap_scores.index,
            text=heatmap_grades.values,
            hovertemplate=(
                "Year: %{x}<br>"
                "Pillar: %{y}<br>"
                "Score: %{z:.1f}<br>"
                "Grade: %{text}<extra></extra>"
            ),
            colorscale='Viridis',
            zmin=1,
            zmax=4,
            colorbar=dict(
                title="PEFA Scores",
                tickvals=list(SCORE_MAPPING.keys()),
                ticktext=list(SCORE_MAPPING.values()),
            ),
        )
    )

    fig.update_layout(
        title={
            'text': 'How did various pillars of the budget institutions change over time?',
            'font': {'size': 16},
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
        },
        xaxis_title='',
        yaxis_title='Pillars',
        yaxis=dict(tickmode='linear', categoryorder="category descending"),
    )
    return fig


def _score_to_grade(score):
    return SCORE_MAPPING[min(SCORE_MAPPING.keys(), key=lambda x: abs(x - score))]
