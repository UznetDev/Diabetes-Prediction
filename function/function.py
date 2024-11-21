import altair as alt
import pandas as pd


def make_donut(input_response, input_text, input_color=None, 
               R=230, innerRadius=75, cornerRadius=20, fontSize=20):
    if input_color is None:
        input_color = 'blue' if input_response < 50 else 'red'

    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    elif input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    elif input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    elif input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']

    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100 - input_response, input_response]
    })
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })

    plot = alt.Chart(source).mark_arc(innerRadius=innerRadius, cornerRadius=cornerRadius).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            domain=[input_text, ''],
                            range=chart_color),
                        legend=None),
    ).properties(width=R, height=R)

    text = plot.mark_text(align='center', 
                          color=chart_color[0], 
                          font="Lato", 
                          fontSize=fontSize, 
                          fontWeight=100,
                          fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=innerRadius, 
                                            cornerRadius=cornerRadius).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            domain=[input_text, ''],
                            range=chart_color),
                        legend=None),
    ).properties(width=R, height=R)

    return plot_bg + plot + text