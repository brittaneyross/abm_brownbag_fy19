
#import libraries
import fiona
import pandas as pd
import numpy as np
from os.path import dirname, join
import os

#bokeh
from bokeh.io import show, output_notebook, push_notebook, curdoc, output_file
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import layout, column, row
from bokeh.models.selections import Selection
from bokeh.models import Select,NumberFormatter,CustomJS, Panel, Spacer,HoverTool,LogColorMapper, ColumnDataSource, TapTool, BoxSelectTool, LabelSet, Label, FactorRange,NumeralTickFormatter
from bokeh.tile_providers import STAMEN_TERRAIN_RETINA,CARTODBPOSITRON_RETINA
from bokeh.core.properties import value
from bokeh.transform import factor_cmap, dodge
from bokeh.models.widgets import Div, Tabs, Paragraph, Dropdown, Button, PreText, Toggle, TableColumn, DataTable

#mapping
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon
import geopandas as gpd

from bokeh.transform import factor_cmap
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.core.properties import value

#color
from bokeh.palettes import Spectral6

import warnings
warnings.filterwarnings('ignore')

from bokeh.io import curdoc

column_width = 1400
margin = 100

dist_bar_values = pd.read_csv(join(dirname(__file__),'data','dist_bar.csv'))
mode_bar_values = pd.read_csv(join(dirname(__file__),'data','mode_bar.csv'))

hh_sample = pd.read_csv(join(dirname(__file__),'data','sample_data','hh_sample.csv'))
per_sample = pd.read_csv(join(dirname(__file__),'data','sample_data','per_sample.csv'))
iTours_sample = pd.read_csv(join(dirname(__file__),'data','sample_data','itour_sample.csv'))
itrips_sample = pd.read_csv(join(dirname(__file__),'data','sample_data','itrips_sample.csv'))
ecd = join(dirname(__file__),'data','shapefiles','ecd_shp.shp')
ecd_maz = join(dirname(__file__),'data','shapefiles','ecdmazdest_shp.shp')
ecd_trips_attr = pd.read_csv(join(dirname(__file__),'data','ecd_trip_groups.csv'))


def make_filter_vbar(df, groups_field, subgroups, filters, tool_tips, chart_tools,palette_color,
                     p_width = 400, p_height = 200, chart_title="Sample Grouped Bar Chart",
                     drop_down_label = "Sample Dropdown"):

    df_copy = df.copy()

    def filter_df(df, fvalue):
        if fvalue:
            return df.loc[df[filters]==fvalue]
        else:
            return df

    def update(attr, old, new):

        drop_value = drop_down.value

        df_copy = filter_df(df, drop_value)

        new_source = make_source(df_copy)[0]
        chart_source.data.update(new_source.data)

        p.title.text = chart_title + "-" + drop_value

    def make_source(df_src):
        df_groupby = df_src.groupby([groups_field]).sum().reset_index()
        df_groups = df_groupby[groups_field].values.tolist()
        numgroups = len(subgroups)

        data = {'groups': df_groups}

        ziplist = ()
        for s in subgroups:
            data[s] = df_groupby[s].values.tolist()
            ziplist += (data[s],)


        x = [(g, s) for g in df_groups for s in subgroups]
        sgroups = [s for g in df_groups for s in subgroups]
        pgroups = [g for g in df_groups for s in subgroups]

        if numgroups == 2:
            counts = sum(zip(ziplist[0], ziplist[1]), ())
        elif numgroups ==3:
            counts = sum(zip(ziplist[0], ziplist[1], ziplist[2]), ())
        elif numgroups ==4:
            counts = sum(zip(ziplist[0], ziplist[1], ziplist[2], ziplist[3]), ())
        elif numgroups ==5:
            counts = sum(zip(ziplist[0], ziplist[1], ziplist[2], ziplist[3], ziplist[4]), ())
        elif numgroups ==6:
            counts = sum(zip(ziplist[0], ziplist[1], ziplist[2], ziplist[3], ziplist[4], ziplist[5]), ())
        elif numgroups ==7:
            counts = sum(zip(ziplist[0], ziplist[1], ziplist[2], ziplist[3], ziplist[4],
                             ziplist[5], ziplist[6]), ())
        elif numgroups ==8:
            counts = sum(zip(ziplist[0], ziplist[1], ziplist[2], ziplist[3], ziplist[4],
                             ziplist[5], ziplist[6], ziplist[7]), ())
        elif numgroups ==9:
            counts = sum(zip(ziplist[0], ziplist[1], ziplist[2], ziplist[3], ziplist[4],
                             ziplist[5], ziplist[6], ziplist[7], ziplist[8]), ())
        elif numgroups ==10:
            counts = sum(zip(ziplist[0], ziplist[1], ziplist[2], ziplist[3], ziplist[4],
                             ziplist[5], ziplist[6], ziplist[7], ziplist[8], ziplist[9]), ())

        source = ColumnDataSource(data=dict(x=x, counts=counts, sub=sgroups, prime=pgroups))

        return source, x, sgroups
    def make_chart(source,x,sgroups):

        p = figure(x_range=FactorRange(*x), plot_width = p_width,plot_height=p_height, title=chart_title,
        toolbar_location='right', tools=chart_tools,
        tooltips=tool_tips)

        p.vbar(x='x', top='counts', width=0.9, source=source,
              line_color='white', fill_color=factor_cmap('x', palette=palette_color, factors=sgroups, start=1, end=2))

        return p

    df_copy = df.copy()

    filter_menu = []
    for f in df[filters].drop_duplicates().values.tolist():
        filter_menu.append((f.title(),f))

    drop_down = Dropdown(label=drop_down_label, button_type="default", menu=filter_menu, width=250)
    drop_down.on_change('value', update)


    src = make_source(df_copy.loc[df_copy['Filter']=='Discretionary'])
    chart_source = src[0]
    x = src[1]
    sgroups = src[2]

    p = make_chart(chart_source, x, sgroups)

    # Styling
    #p = bar_style(p)

    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None

    return column(drop_down, p)
        #return source

def make_base_map(tile_map=CARTODBPOSITRON_RETINA,map_width=800,map_height=500, xaxis=None, yaxis=None,
                xrange=(-9990000,-9619944), yrange=(5011119,5310000),plot_tools="pan,wheel_zoom,reset"):

    p = figure(tools=plot_tools, width=map_width,height=map_height, x_axis_location=xaxis, y_axis_location=yaxis,
                x_range=xrange, y_range=yrange)

    p.grid.grid_line_color = None

    p.add_tile(tile_map)

    return p
def make_poly_map(base_map, shapefile,label,fillcolor,fillalpha,linecolor,lineweight,add_label,legend_field):

    p = base_map

    shp = fiona.open(shapefile)

    # Extract features from shapefile
    district_name = [ feat["properties"][label].replace(" County","") for feat in shp]
    pareas = [ feat["properties"][legend_field] for feat in shp]
    pop = [ feat["properties"]["TOT_POP"] for feat in shp]
    district_area = [ feat["properties"]["Shape_Area"] for feat in shp]
    district_x = [ [x[0] for x in feat["geometry"]["coordinates"][0]] for feat in shp]
    district_y = [ [y[1] for y in feat["geometry"]["coordinates"][0]] for feat in shp]
    district_xy = [ [ xy for xy in feat["geometry"]["coordinates"][0]] for feat in shp]
    district_poly = [ Polygon(xy) for xy in district_xy] # coords to Polygon

    source = ColumnDataSource(data=dict(
        x=district_x, y=district_y,
        name=district_name,
        planning = pareas,
        pop=pop
    ))

    ecd_df = pd.DataFrame({'x':district_x,'y':district_y,'name':district_name,'planning':pareas,'pop':pop})

    #ecd_df.to_csv(os.path.join(cur,'abm_pres','data','ecd_src.csv'), index=False)

    polygons = p.patches('x', 'y', source=source, fill_color=fillcolor,
              fill_alpha=fillalpha, line_color=linecolor, line_width=lineweight, legend=legend_field)

    if add_label:

        labels = LabelSet(x='label_x', y='label_y', source=source,text='name', level='glyph',text_line_height=1.5,
                  x_offset = -15,y_offset = -8,render_mode='canvas',text_font_size="10pt",text_color="white")

        p.add_layout(labels)

    TOOLTIPS = [
        ("Census Tract", '@name'),
        ("Total Population", '@pop')
    ]

    p.add_tools(HoverTool(tooltips=TOOLTIPS, renderers=[polygons]))


    return p

def load_image(image, title_text):
    p = figure(x_range=(0,250), y_range=(0,410),plot_width=600, plot_height=800,
               x_axis_location=None, y_axis_location=None,tools='',title=title_text)
    p.image_url(url=[image], x=0, y=1, w=250, h=410, anchor="bottom_left")
    return p

def abm_background():
    b_title = Div(text = """<h1>What is an Activity Based Model (ABM)?</h1><br>""",width = column_width)
    background = Div(text = """<ul><li><h3>Travel demand model that simulates  individual and household transportation decisions</h3></li><br>
                            <li><h3>Generates activities, identifies destinations for activities, determines mode of travel, and assigns routes on our network</h3></li><br>
                            <li><h3>Considers personal and household attributes to predict:</h3></li>
                                <ul><li>Types of activities a traveler participates in</li>
                                <li>The individual and/or household members participating in the activity</li>
                                <li>Where to participate in the activity</li>
                                <li>How activities are scheduled/prioritized</li>
                                <li>Time available to participate in those activities</li>
                                <li>Mode of travel reach each activity</li>
                                </ul><br>
                            <li><h3>Produces a behaviorally realistic representation of travel compared to the trip-based model</h3></li>
                            </ul>
                            """,width=int(column_width*.4), style={'font-size':'150%'})

    model_tbl = pd.DataFrame({'Travel Questions': ['What activities do people want to participate in?',
                                                  'Where are these activities?','When are these activities?',
                                                  'What travel mode is used?','What route is used?'],
                             'Trip-Based Model': ['Trip generation','Trip distribution','None','Trip mode choice','Network assignment'],
                             'Activity-Based Model': ['Activity generation and scheduling','Tour and trip destination choice',
                                                       'Tour and trip time of day','Tour and trip mode choice','Network assignment']})

    div_tbl = Div(text=model_tbl.to_html(index=False,classes=["table-bordered", "table-hover"],),
                    width=int(column_width*.6), height = 500, style={'font-size':'120%'})

    return row(Spacer(width = margin), column(Spacer(height=25),b_title,row(background, Spacer(width=int(column_width*.10)), column(Spacer(height=80),div_tbl))))

def key_features():

    kf_title = Div(text="""<h1>Why use an activity based model?</h1>""", width = column_width)

    kf_attr = Div(text="""<h3>Each Traveler's Personal Attributes Inform Unique Travel Choices</h3>
                            <ul><li>Makes the model more sensitive to planning strategies and policies that:</li>
                           <ul><li>Influence how an individual budgets time for activities and travel throughout the day </li>
                               <li>Promote carpooling, increase/decrease transit access/cost, implement tolling, etc</li>
                           </ul></ul>
                       """,width=int(column_width*.6), style={'font-size':'150%'})

    attr_tbl = pd.DataFrame({'Personal Attributes': ['Age',
                                                  'Employment Status',
                                                  'School Status'],
                             'Household Attributes': ['Income', 'Number of Vehicles','Household Size']})

    div_attr_tbl = Div(text=attr_tbl.to_html(index=False,classes=["table-bordered", "table-hover"],), style={'font-size':'120%'})

    kf_traveler = Div(text="""<h3>Each Traveler Can Be Identified Throughout Simulation</h3>
                                <ul><li>Works at the disaggregate person level rather than the aggregate zonal-level</li>
                                <li>Enables analysis of travel patterns across a wide range of dimensions (socioeconomic groups)</li>
                                <li>Useful in developing metrics on how travel benefits or disbenefits impact different populations </li>
                                </ul>
                       """,width=int(column_width*.6), style={'font-size':'150%'})



    purp_bar_chart = make_filter_vbar(dist_bar_values, 'Distance', ['0-35K', '35K-60k','60K-100K', '100K+'], "Filter",
                     [("label1"," @prime"), ("label2", "@sub"), ("label3", "@counts{0.0f}%")], 'hover',Spectral6,
                     p_width = 600, p_height = 300, chart_title="Distance by Activity and Income",
                     drop_down_label = "Choose Trip Purpose")

    mode_bar_chart = make_filter_vbar(mode_bar_values, 'trip_mode', ['0-35K', '35K-60k','60K-100K', '100K+'], "Filter",
                     [("label1"," @prime"), ("label2", "@sub"), ("label3", "@counts{0.0f}%")], 'hover',Spectral6,
                     p_width = 600, p_height = 300, chart_title="Distance by Activity and Income",
                     drop_down_label = "Choose Trip Purpose")



    kf_tours = Div(text="""<h3>Travel Is Organized Into Tours and Trips</h3>
                            <ul><li>Travel for each individual or group is represented in a tour, a chain of trips that start and end at home, to generate daily activity pattern</li>
                            <li>Trips making up a tour portray how activities are organized as well as modal decisions made by the person during the trip</li>
                            <li>Enables segmentation by trip purpose (activity) and mode</li>
                            </ul>
                       """, width=int(column_width*.6), style={'font-size':'150%'})

    return (row(Spacer(width = margin), column(Spacer(height=25),
            row(column(kf_title, kf_attr, Spacer(height=150), kf_traveler, Spacer(height=100), kf_tours,width=int(column_width*.6)),Spacer(width=10),
            column(Spacer(height=50),div_attr_tbl, Spacer(height=20),purp_bar_chart,Spacer(height=20),mode_bar_chart)))))


def overview_tab():
    #image linke
    #image = load_image(abm_flow,'CT-RAMP Structure and Sub-Models')
    flow_image = Div(text="<img src='abm_pres/static/images/abm_flow_chart.png'>")

    overview_text = Div(text="""<h1>Activity Based Model Overview</h1>""", width = column_width)
    ctramp_text = Div(text="""<h3>Coordinated Travel - Regional Activity Modeling Platform
                        (CT-RAMP) </h3><p>ABM model implements the CT-RAMP design and software platform.
                        Features microsimulation of travel decisions for individual households and persons
                        within a household as well as intra-household interactions
                        across a range of activity and travel dimensions.</p><br>
                        <ol><li><b>Population synthesis</b> creates and distributes households and persons
                        for use in the ABM</li><br>
                        <li><b>Long-Term Location Choice</b> - Models location of usual (mandatory) destinations</li><br>
                        <li><b>Mobility</b> - Models household attributes like free parking eligibility, car ownership,
                        transit pass, or toll transponder</li><br>
                        <li><b>Coordinated Daily Activity-Travel Pattern</b> - Generates and schedules mandatory
                        and non-mandatory activities for each household member.</li><br>
                        <li><b>Tour</b> - Daily activities are organized into tours, tour mode, number,
                        and location of stops are determined.</li><br>
                        <li><b>Trips</b> - Mode, parking, and location of trips making up tour is determined.</li><br>
                        <li><b>Network Simulation</b> - List of trips for each individual and/or travel party
                        is generated and trips routes are assigned on the modeling network for auto and transit.</li>
                        </ol>""", width = int(column_width*.5), style={'font-size':'150%'})
    extra = Div(text="""<hr><ul><li>Tour: Chain of trips that start and end at home</li>
                        <li>Person Types: 8 Person Types (1=Full time worker, 2=Part time worker, 3=University student,
                        4=Adult non-worker under 65, 5=retiree, 6=driving age school child, 7=pre-driving age school child, 8=preschool child)</li>
                        <li>Activities: 10 travel purposes (work, university, school, escorting,
                        shopping, other maintenance, eating out, visiting relatives and friends,
                        other discretionary, and atwork)</li>
                        <li>Modes = 21 modes at both tour and trip level</li>
                        </ul>""", width = int(column_width*.5),
                        css_classes = ['small'])

    return (row(Spacer(width = margin), (column(Spacer(width = margin, height=25),overview_text,row(column(ctramp_text,extra),Spacer(width=100),column(flow_image))))))


def output_tab():

    hh_col = [TableColumn(field=col, title=col) for col in hh_sample.columns[:2]]+\
             [TableColumn(field='income', title='income',formatter=NumberFormatter(format="$0,0.00"))]+\
             [TableColumn(field=col, title=col) for col in hh_sample.columns[3:]]

    per_col = [TableColumn(field=col, title=col) for col in per_sample.columns]
    tour_col = [TableColumn(field=col, title=col) for col in iTours_sample.columns]
    trip_col = [TableColumn(field=col, title=col) for col in itrips_sample.columns]

    hh_src = ColumnDataSource(hh_sample.sort_values(by='hh_id'))
    per_src = ColumnDataSource(per_sample.sort_values(by='hh_id'))
    tour_src = ColumnDataSource(iTours_sample.sort_values(by='person_id'))
    trip_src = ColumnDataSource(itrips_sample.sort_values(by='person_id'))


    hh_div = Div(text="""<h3>Household Attribution Results</h3><p>Individual household attributes</p>
                        <ul><li><b>hh_id</b> : Household ID</li>
                        <li><b>maz</b> : Household Subzone</li>
                        <li><b>income</b> : Household Income</li>
                        <li><b>autos</b> : Number of Vehicles</li>
                        <li><b>size</b> : Household Size</li>
                        <li><b>workers</b> : Number of Workers in Household</li>
                        <li><b>auto_suff</b> : Auto Sufficiency</li></ul>""", width=int(column_width*.45))
    hh_tbl = DataTable(columns=hh_col, source=hh_src, height = 200, selectable = True,width=int(column_width*.45),
                             fit_columns = True, scroll_to_selection=True)

    per_div = Div(text="""<h3>Person Attribution Results</h3><p>Individual persons within a household</p>
                        <ul><li><b>hh_id</b> : Household ID</li>
                        <li><b>person_id</b> : Person ID</li>
                        <li><b>per_num</b> : Person Number in Household</li>
                        <li><b>age</b> : Age</li>
                        <li><b>gender</b> : Gender</li>
                        <li><b>type</b> : Person Type (worker, student, etc)</li></ul>""", width=int(column_width*.45))
    per_tbl = DataTable(columns=per_col, source=per_src,height = 200,selectable = True,width=int(column_width*.45),
                         fit_columns = True, scroll_to_selection=True)

    tour_div = Div(text="""<h3>Tour Results</h3><p>Tours by person and household</p>
                        <ul><li><b>hh_id</b> : Household ID</li>
                        <li><b>person_id</b> : Person ID</li>
                        <li><b>tour_id</b> : Tour ID (0=first tour, 1 second tour, ect)</li>
                        <li><b>tour_category</b> : Mandatory, Non-Mandatory</li>
                        <li><b>tour_purpose</b> : Purpose of travel</li>
                        <li><b>maz</b> : Origin and destination subzone</li>
                        <li><b>tour_mode</b> : Mode of travel</li></ul>""", width=int(column_width*.4))
    tour_tbl = DataTable(columns=tour_col, source=tour_src,height = 250,selectable = True,width=int(column_width*.45),
                         fit_columns = True, scroll_to_selection=True)

    line_div = trip_div = Div(text="""<hr>""", width = column_width)
    trip_div = Div(text="""<h3>Trip Results</h3><p>Trips by person and household</p>
                        <ul><li><b>hh_id</b> : Household ID</li>
                        <li><b>person_id</b> : Person ID</li>
                        <li><b>tour_id</b> : Tour ID (0=first tour, 1 second tour, ect)</li>
                        <li><b>purpose</b> : Origin and destination trip purpose</li>
                        <li><b>maz</b> : Origin and destination subzone</li>
                        <li><b>trip_mode</b> : Mode of travel</li>
                        <li><b>tap</b> : boarding and alighting transit id</li></ul>""", width=int(column_width*.45))
    trip_tbl = DataTable(columns=trip_col, source=trip_src,height = 250,selectable = True,width=int(column_width*.45),
                         fit_columns = True, scroll_to_selection=True)

    def hh_select():

        indices = hh_src.selected.indices
        if len(indices) == 1:
            hh_id = hh_src.data["hh_id"][indices[0]]

            new_indices = [i for i, h in enumerate(per_src.data["hh_id"]) if h == hh_id]
            per_src.selected.indices=new_indices


    #hh_src.on_change('selected',my_tap_handler)

    per_button = Button(label="Select Person Level", button_type="default")
    per_button.on_click(hh_select)


    def per_select():

        indices = per_src.selected.indices
        if len(indices) == 1:
            per_id = per_src.data["person_id"][indices[0]]

            new_indices = [i for i, p in enumerate(tour_src.data["person_id"]) if p == per_id]
            tour_src.selected.indices=new_indices


    tour_button = Button(label="Select Person Tour", button_type="default")
    tour_button.on_click(per_select)


    def trip_select():

        indices = tour_src.selected.indices
        if len(indices) == 1:
            per_id = tour_src.data["person_id"][indices[0]]
            tour_id = tour_src.data["tour_id"][indices[0]]


            new_indices = []
            i = 0
            while i < len(trip_src.data["person_id"]):
                #trip_src.data["person_id"][i] == per_id
                if trip_src.data["person_id"][i] == per_id:
                    if trip_src.data["tour_id"][i] == tour_id:
                        new_indices.append(i)
                i+=1

            trip_src.selected.indices=new_indices

    trip_button = Button(label="Select Person Trip", button_type="default")
    trip_button.on_click(trip_select)

    output_title = Div(text = """<h1>Model Output Files</h1>""",width=column_width)
    output_description = Div(text = """<p>The ABM produces output files for modeled householdes, modeled persons,
    mandatory trip locations (work, school, university), trips, tours, ect. Model data calibrated to the
    CMAP Household Travel Survey (2007-2009)<p>""",width=column_width)

    #return row(Spacer(width = margin),column(Spacer(height=25),output_title,output_description, hh_div, hh_tbl,  per_div, per_button,per_tbl,
    #             Spacer(height=10),tour_div,tour_button, tour_tbl, Spacer(height=10),trip_div,trip_button, trip_tbl))

    return (row(Spacer(width = margin), column(Spacer(height=25),output_title,output_description,
            row(column(hh_div,hh_tbl,per_button), Spacer(width = int(column_width*.1)), column(per_div,per_tbl,tour_button)),line_div,
            row(column(tour_div,tour_tbl,trip_button), Spacer(width = int(column_width*.1)), column(trip_div,trip_tbl)))))


def data_explore():
    d_title = Div(text="""<h1>Trips by Activity</h1>""", width = column_width)

    #develop map data_explore

    total_trips = pd.crosstab(index = ecd_trips_attr['destsubzone09'],columns=ecd_trips_attr['trip_purpose'],
            values=ecd_trips_attr['Model'],aggfunc=sum).fillna(0).reset_index()

    shp = fiona.open(ecd_maz)
    mazs = [y_val["properties"]["subzone09"] if y_val["properties"]["subzone09"] <= 16443 else 0 for y_val in shp]

    base = pd.DataFrame({'subzones': mazs})
    cmap = base.loc[base['subzones'] > 0].sort_values(by='subzones')

    cmap_attr = cmap.merge(total_trips, how='left', left_on = 'subzones', right_on = 'destsubzone09').fillna(0)

    #cmap_attr.to_csv(os.path.join(cur,'abm_pres','data','cmap_purp.csv'), index=False)

    cmap_attr = cmap_attr.set_index('subzones')

    trips_dict = cmap_attr.to_dict()

    max_subzone = 16443
    shp = fiona.open(ecd_maz)
    #subzones = [y_val["properties"]["subzone09"] if y_val["properties"]["subzone09"]  <= max_zone else  for y_val in shp]
    subzones = []
    district_x = []
    district_y = []
    district_xy = []
    district_poly = []

    #district_area = [ feat["properties"]["Shape_Area"] for feat in shp]
    #district_x = [ [x[0] for x in feat["geometry"]["coordinates"][0]] for feat in shp]
    #district_y = [ [y[1] for y in feat["geometry"]["coordinates"][0]] for feat in shp]
    #district_xy = [ [ xy for xy in feat["geometry"]["coordinates"][0]] for feat in shp]
    #district_poly = [ Polygon(xy) for xy in district_xy] # coords to Polygon

    for maz in shp:
        if maz["properties"]["subzone09"] <= max_subzone and maz["properties"]["subzone09"] > 0:
            subzones.append(maz["properties"]["subzone09"])
            district_x.append([x[0] for x in maz["geometry"]["coordinates"][0]])
            district_y.append([y[1] for y in maz["geometry"]["coordinates"][0]])
            district_xy.append([ xy for xy in maz["geometry"]["coordinates"][0]])
            district_xy.append(Polygon(xy) for xy in district_xy)


    disc = []
    eat = []
    esc = []
    main = []
    sch = []
    shop = []
    uni = []
    vis = []
    work = []
    wb = []
    for k,v in trips_dict.items():

        if k == 'Discretionary':
            disc = [trips_dict[k][y_val] for y_val in subzones]
        if k == 'Eating Out':
            eat = [trips_dict[k][y_val]  for y_val in subzones]
        if k == 'Escort':
            esc = [trips_dict[k][y_val]  for y_val in subzones]
        if k == 'Maintenance':
            main = [trips_dict[k][y_val]  for y_val in subzones]
        if k == 'School':
            sch = [trips_dict[k][y_val]  for y_val in subzones]
        if k == 'Shopping':
            shop = [trips_dict[k][y_val]  for y_val in subzones]
        if k == 'University':
            uni = [trips_dict[k][y_val]  for y_val in subzones]
        if k == 'Visiting':
            vis = [trips_dict[k][y_val]  for y_val in subzones]
        if k == 'Work':
            work = [trips_dict[k][y_val]  for y_val in subzones]
        if k == 'Work-Based':
            wb = [trips_dict[k][y_val]  for y_val in subzones]


    cmap_map_src = pd.DataFrame({'subzones':subzones,
    'x':district_x, 'y':district_y,'disc':disc,
    'escort':esc,
    'main':main,
    'school':sch,
    'shop':shop,
    'university':uni,
    'visit':vis,
    'work':work,
    'work-based':wb })


    source = ColumnDataSource(cmap_map_src)

    #cmap_map_src.to_csv(os.path.join(cur,'abm_pres','data','cmap_src.csv'), index=False)

    def make_base_map(tile_map=CARTODBPOSITRON_RETINA,map_width=1000,map_height=800, xaxis=None, yaxis=None,
                    xrange=(-9990000,-9619944), yrange=(5011119,5310000),plot_tools="pan,wheel_zoom,reset"):

        p = figure(tools=plot_tools, width=map_width,height=map_height, x_axis_location=xaxis, y_axis_location=yaxis,
                    x_range=xrange, y_range=yrange)

        p.grid.grid_line_color = None

        p.add_tile(tile_map)

        return p

    p = make_base_map()
    custom_colors = ['#e5f5f9',
    '#99d8c9',
    '#2ca25f']

    color_mapper = LogColorMapper(palette=custom_colors)

    def update(attr, old, new):
        poly_val = drop_down.value

        polygons.glyph.fill_color['field'] = poly_val

        print(polygons.glyph.fill_color['field'])




    polygons = p.patches('x', 'y', source=source, fill_color={'field': 'work', 'transform': color_mapper},
                  fill_alpha=1, line_color=None, line_width=.5)

    poly_plot = make_poly_map(p, ecd, 'NAME',None,.5,'Black',.5,False,"EDA_FLAG")



    filter_menu = [('escort','escort'),('main','main'),('school','school'),('work','work')]

    #drop_down = Dropdown(label='Choose Trip Purpose', button_type="default",
    #                     menu=filter_menu, width=250,default_value='work')
    cb_cselect = CustomJS(args=dict(cir=polygons,csource=source), code ="""
        var selected_color = cb_obj.value;
        cir.glyph.fill_color.field = selected_color;
        csource.change.emit();
        cir.change.emit();
    """)


    select_map = Select(options=['escort','main','school','shop','university','visit','work','work-based'],
                       callback=cb_cselect)

    #drop_down.on_change('value', cb_cselect)


    l=column(select_map,p)

    #table data_explore
    age_groups = pd.crosstab(index = ecd_trips_attr['Age Range'],columns=ecd_trips_attr['trip_mode'],
            values=ecd_trips_attr['Model'],aggfunc=sum).fillna(0)


    age_groups["Total"] = age_groups.sum(axis=1)

    age_groups_per = (age_groups.loc[:,"Bike":"Walk"].div(age_groups["Total"], axis=0)*100).reset_index()

    income = pd.crosstab(index = ecd_trips_attr['hhincome'],columns=ecd_trips_attr['trip_mode'],
                values=ecd_trips_attr['Model'],aggfunc=sum).fillna(0)
    income["Total"] = income.sum(axis=1)
    income_groups_per = (income.loc[:,"Bike":"Walk"].div(income["Total"], axis=0)*100).reset_index()

    float_format='{:20,.1f}%'.format

    age_div = Div(text=age_groups_per.to_html(index=False,float_format='{:20,.1f}%'.format,
    classes=["table-bordered", "table-hover","text-center","table-condensed","thead-dark"]))

    income_div = Div(text=income_groups_per.to_html(index=False,float_format='{:20,.1f}%'.format,
    classes=["table-bordered", "table-hover","text-center","table-condensed","thead-dark"]))

    return row(column(select_map,p),Spacer(width = 50),column(Spacer(height=50),age_div,income_div))




h_1 = Div(text = """<h1><center>Intro Text</center></h1>""",width=column_width)
h_2 = Div(text = """<h1><center>Intro Text</center></h1>""")
h_4 = Div(text = """<h1><center>Intro Text</center></h1>""")

b_0 = layout(children=[abm_background()])
b_1 = layout(children=[key_features()])

l_1 = layout(children=[overview_tab()])
l_2 = layout(children=[output_tab()])
l_3 = layout(children=[data_explore()])

tab_0 = Panel(child=b_0, title ='Background')
tab_1 = Panel(child=b_1, title ='Advantages')

tab_2 = Panel(child=l_1, title ='Model Overview')
tab_3 = Panel(child=l_2, title ='Outputs')
tab_4 = Panel(child=l_3, title ='Data Exploration')

tabs = Tabs(tabs = [tab_0, tab_1, tab_2, tab_3, tab_4], sizing_mode = "stretch_both")

curdoc().add_root(tabs)
