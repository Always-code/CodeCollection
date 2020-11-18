#Radhika Date
#ISE 535 Python Programming for Industrial Engineers - Fall 2020 Final Project
#Statistics and Data Visualization for COVID 19

#START OF PROGRAM

#IMPORTING LIBRARIES
import dash
import dash_core_components as dcc
import dash_table
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math

#####################################################################################################
#####################################################################################################
#FUNCTION DEFINITIONS
def ceilvalue(contlist):
    cl = []
    for i in contlist:
        j = math.ceil(i)
        cl.append(j)
    return(cl)
    
def table_2list(listA):
    listA1 =[]
    for s in range(listA.shape[0]):
        listA1.append(listA.iloc[s][0])
    return(listA1)

#READING DATA FILE AND PREPARING IT FOR ANALYSIS
pd.set_option('display.max_columns',10)
rawcovid = pd.read_csv('D:\\Third sem academic\\ISE 535\\project\\COVID-19 Activity.csv',parse_dates=['REPORT_DATE'])
covid = rawcovid.rename(columns = {"PEOPLE_POSITIVE_CASES_COUNT":"positive_cases","COUNTY_NAME":"county","PROVINCE_STATE_NAME":"state","REPORT_DATE":"report_date","CONTINENT_NAME":"continent","DATA_SOURCE_NAME":"data_source","PEOPLE_DEATH_NEW_COUNT":"new_deathcount","COUNTY_FIPS_NUMBER":"fips","COUNTRY_ALPHA_3_CODE":"alpha3","COUNTRY_SHORT_NAME":"country","COUNTRY_ALPHA_2_CODE":"alpha2","PEOPLE_POSITIVE_NEW_CASES_COUNT":"new_positivecasecount","PEOPLE_DEATH_COUNT":"deathcount"})
covid.index.name = 'serial_number'
countnan = covid.isna().sum()
c1 = covid.drop(labels=covid.columns[14],axis=1)
covidfinal = c1.drop(labels=covid.columns[13],axis=1)

#DATA FRAMES AND CALCULATION
countyisnull = covidfinal[covidfinal['county'].isnull()]
covidfinal1 = countyisnull.groupby(['continent','country','state'])
table_1 = pd.DataFrame(columns = ['Continent','Country','State','Min positive cases','Max positive cases','Avg positive cases','Min deaths','Max deaths','Avg deaths'])
for key,item in covidfinal1:
    groupeddata_1 = covidfinal1.get_group(key)
    minpositivecase = groupeddata_1['positive_cases'].min()
    maxpositivecase = groupeddata_1['positive_cases'].max()
    mindeathcount = groupeddata_1['deathcount'].min()
    maxdeathcount = groupeddata_1['deathcount'].max()
    avgpositivecases = groupeddata_1['positive_cases'].mean()
    avgdeathcount = groupeddata_1['deathcount'].mean()
    table_1 = table_1.append({'Continent':groupeddata_1['continent'].unique(),'Country':groupeddata_1['country'].unique(),'State':groupeddata_1['state'].unique(),'Min positive cases':minpositivecase,'Max positive cases':maxpositivecase,'Avg positive cases':avgpositivecases,'Min deaths':mindeathcount,'Max deaths':maxdeathcount,'Avg deaths':avgdeathcount},ignore_index=True)
dec = 2
table_1['Avg positive cases'] = table_1['Avg positive cases'].apply(lambda x: round(x,dec))
table_1['Avg deaths'] = table_1['Avg deaths'].apply(lambda x: round(x,dec))


stateisnull = covidfinal[covidfinal['state'].isnull()]
covidfinal2 = stateisnull.groupby(['continent','country'])
table_2 = pd.DataFrame(columns = ['Continent','Country','Min positive cases','Max positive cases','Avg positive cases','Min deaths','Max deaths','Avg deaths'])
for key,item in covidfinal2:
    groupeddata_2 = covidfinal2.get_group(key)
    minpositivecase = groupeddata_2['positive_cases'].min()
    maxpositivecase = groupeddata_2['positive_cases'].max()
    mindeathcount = groupeddata_2['deathcount'].min()
    maxdeathcount = groupeddata_2['deathcount'].max()
    avgpositivecases = groupeddata_2['positive_cases'].mean()
    avgdeathcount = groupeddata_2['deathcount'].mean()
    table_2 = table_2.append({'Continent':groupeddata_2['continent'].unique(),'Country':groupeddata_2['country'].unique(),'Min positive cases':minpositivecase,'Max positive cases':maxpositivecase,'Avg positive cases':avgpositivecases,'Min deaths':mindeathcount,'Max deaths':maxdeathcount,'Avg deaths':avgdeathcount},ignore_index=True)
dec = 2
table_2['Avg positive cases'] = table_2['Avg positive cases'].apply(lambda x: round(x,dec))
table_2['Avg deaths'] = table_2['Avg deaths'].apply(lambda x: round(x,dec))
listAf = table_2.loc[table_2['Continent'] == 'Africa','Country']
listA1 = table_2list(listAf)
listAf = table_2.loc[table_2['Continent'] == 'Europe','Country']
listA2 = table_2list(listAf)
listAf = table_2.loc[table_2['Continent'] == 'Asia','Country']
listA3 = table_2list(listAf)
listAf = table_2.loc[table_2['Continent'] == 'America','Country']
listA4 = table_2list(listAf)
listAf = table_2.loc[table_2['Continent'] == 'Oceania','Country']
listA5 = table_2list(listAf)

lblstate = []
countynotnull = covidfinal[covidfinal.county.notnull()]
covidfinal3 = countynotnull.groupby(['continent','country','state','county'])
table_3 = pd.DataFrame(columns = ['Continent','Country','State','County','Min positive cases','Max positive cases','Avg positive cases','Min deaths','Max deaths','Avg deaths'])
for key,item in covidfinal3:
    groupeddata_3 = covidfinal3.get_group(key)
    minpositivecase = groupeddata_3['positive_cases'].min()
    maxpositivecase = groupeddata_3['positive_cases'].max()
    mindeathcount = groupeddata_3['deathcount'].min()
    maxdeathcount = groupeddata_3['deathcount'].max()
    avgpositivecases = groupeddata_3['positive_cases'].mean()
    avgdeathcount = groupeddata_3['deathcount'].mean()
    #print(minpositivecase,maxpositivecase,maxdeathcount,avgpositivecases,avgdeathcount)
    table_3 = table_3.append({'Continent':groupeddata_3['continent'].unique(),'Country':groupeddata_3['country'].unique(),'State':groupeddata_3['state'].unique(),'County':groupeddata_3['county'].unique(),'Min positive cases':minpositivecase,'Max positive cases':maxpositivecase,'Avg positive cases':avgpositivecases,'Min deaths':mindeathcount,'Max deaths':maxdeathcount,'Avg deaths':avgdeathcount},ignore_index=True)
    if groupeddata_3['state'].unique() not in lblstate:
        lblstate.append(groupeddata_3['state'].unique()[0])
dec = 2
table_3['Avg positive cases'] = table_3['Avg positive cases'].apply(lambda x: round(x,dec))
table_3['Avg deaths'] = table_3['Avg deaths'].apply(lambda x: round(x,dec))
fl = []
for v in lblstate:
    new1 = table_3.loc[table_3['State'] == v,'County']
    new2 = table_2list(new1)
    fl.append(new2)

labels = []
contpossum = []
contlist = []
covidfinal4 = covidfinal.groupby(['continent'])
table_4 = pd.DataFrame(columns = ['Continent','Country1','Max positive cases','Country2','Max deaths'])
for key4,item in covidfinal4:
    groupeddata_4 = covidfinal4.get_group(key4)
    contlist.append(key4)
    labels.append(groupeddata_4['country'].unique())
    sumpositivecases = groupeddata_4['positive_cases'].sum()
    contpossum.append(sumpositivecases)
    maxdeaths = groupeddata_4['deathcount'].max()
    maxpositivecase = groupeddata_4['positive_cases'].max()
    countrym = groupeddata_4.loc[groupeddata_4['deathcount']== maxdeaths,'country'].iloc[0]
    countrymaxname = groupeddata_4.loc[groupeddata_4['positive_cases']==maxpositivecase,'country'].iloc[0]
    table_4 = table_4.append({'Continent':groupeddata_4['continent'].unique(),'Country1':countrymaxname,'Max positive cases':maxpositivecase,'Country2':countrym,'Max deaths':maxdeaths},ignore_index = True)

covidfinal5 = covidfinal.groupby(['continent','country'])
africalistp = []
africalistd = []
americalistp = []
americalistd = []
europelistp = []
europelistd = []
oceanslistp = []
oceanslistd = []
asialistp = []
asialistd = []
for key,item in covidfinal5:
    groupeddata_5 = covidfinal5.get_group(key)
    meanpositivecases = groupeddata_5['positive_cases'].mean()
    meandeaths = groupeddata_5['deathcount'].mean()
    if key[0] == 'Africa':
        africalistp.append(meanpositivecases)
        africalistd.append(meandeaths)
    if key[0] == 'America':
        americalistp.append(meanpositivecases)
        americalistd.append(meandeaths)
    if key[0] == 'Europe':
        europelistp.append(meanpositivecases)
        europelistd.append(meandeaths)
    if key[0] == 'Asia':
        asialistp.append(meanpositivecases)
        asialistd.append(meandeaths)
    if key[0] == 'Oceania':
        oceanslistp.append(meanpositivecases)
        oceanslistd.append(meandeaths)
africalistp = ceilvalue(africalistp)
africalistd = ceilvalue(africalistd)
americalistp = ceilvalue(americalistp)
americalistd = ceilvalue(americalistd)
europelistp = ceilvalue(europelistp)
europelistd = ceilvalue(europelistd)
asialistp = ceilvalue(asialistp)
asialistd = ceilvalue(asialistd)
oceanslistp = ceilvalue(oceanslistp)
oceanslistd = ceilvalue(oceanslistd)
label1 = list(labels[0])
label2 = list(labels[1])
label3 = list(labels[2])
label4 = list(labels[3])
label5 = list(labels[4])

covidfinal6 = covidfinal.groupby(['continent','report_date'])
table_7 = pd.DataFrame(columns = ['Continent','Month','Total positive cases'])
for key5,item in covidfinal6:
    gd = covidfinal6.get_group(key5)
    sp = gd['positive_cases'].sum()
    table_7 = table_7.append({'Continent':gd['continent'].unique()[0],'Month':key5[1],'Total positive cases':sp},ignore_index = True)
fig = px.line(table_7,x = 'Month',y = 'Total positive cases',title = 'Total positive cases v/s time',color = 'Continent',hover_name = 'Continent')
fig.update_layout(font_color = '#FFFFFF', title_font_color = '#FFFFFF', title_font_size = 13,title_xref = 'paper',title_x = 0.5,margin = dict(l=20,r=20,b=30,t=30),paper_bgcolor = '#022C6A')
fig.update_xaxes(title_font=dict(size=12, color='#FFFFFF'))
fig.update_yaxes(title_font=dict(size=12, color='#FFFFFF'))
    
el = []
afl = []
asl = []
aml = []
ocl = []
table_5 = pd.DataFrame(columns = ['continent','country','growth_rate','per_growth_rate'])
for key,item in covidfinal5:
    groupeddata_6 = covidfinal5.get_group(key)
    groupeddata_6['month'] = groupeddata_6['report_date'].dt.month
    groupeddata_7 = groupeddata_6.groupby(['month'])
    for key2,item2 in groupeddata_7:
        if key2 == 3:
            groupeddata_8 = groupeddata_7.get_group(key2)
            summarch = groupeddata_8['new_positivecasecount'].sum()
            if summarch == 0:
                summarch = 1
        if key2 == 9:
            groupeddata_8 = groupeddata_7.get_group(key2)
            sumsept = groupeddata_8['new_positivecasecount'].sum()
            gr = ((sumsept/summarch)**(1/7))-1 
            grp = gr*100
    table_5 = table_5.append({'continent':key[0],'country':key[1],'growth_rate':gr,'per_growth_rate':grp},ignore_index = True)
table_6 = table_5.replace([np.inf, -np.inf], np.nan).dropna(axis = 0) 
dec = 2
table_6['per_growth_rate'] = table_6['per_growth_rate'].apply(lambda x: round(x,dec))
el = table_6.loc[table_6['continent'] == 'Europe','country'].tolist()
afl = table_6.loc[table_6['continent'] == 'Africa','country'].tolist()
aml = table_6.loc[table_6['continent'] == 'America','country'].tolist()
ocl = table_6.loc[table_6['continent'] == 'Oceania','country'].tolist()
asl = table_6.loc[table_6['continent'] == 'Asia','country'].tolist()

"""
#FUTURE ENHANCEMENT TO BE ADDED IN DASHBOARD:
ll = []
for i in contpossum:
    ll.append(str(i)+' positive cases')
fig = plt.figure()
ax = fig.subplots()
wedges, text = ax.pie(contpossum, wedgeprops = dict(width = 0.5), startangle = -40)
kw = dict(arrowprops = dict(arrowstyle = '-'),bbox = dict(boxstyle = 'square,pad = 0.3',fc = 'w',ec = 'k',lw = 0.72),zorder = 0, va = 'center')
 
for i,p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    ha = {-1:'right',1:'left'}[int(np.sign(x))]
    cs = f'angle,angleA=0,angleB={ang}'
    kw['arrowprops'].update({'connectionstyle':cs})
    ax.annotate(ll[i],xy = (x,y),xytext = (1.35*np.sign(x),1.4*y),horizontalalignment = ha, **kw)
ax.set_title('Positive cases per continent')
ax.legend(contlist,loc = 'center')
plt.show()
"""

#####################################################################################################
#####################################################################################################

#DASHBOARD LAYOUT
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)

#TEXT AND COLOUR SETTINGS
colors = {'background':'#022C6A','text':'#FFFFFF'}
introduction = """COVID-19 data from Dec 31,2019 to Oct 8,2020 was obtained. 
Total positive cases, positive cases per day and death count values were noted for a hierarychy of
geographical regions all over the globe: Continents - Countries - States - Counties. This dashboard presents statistical and 
graphical analysis of COVID-19 data in varying combinations based on the availability of data.The layout of the dashboard is as follows:"""
m1 = """Section I - Statistics for Canada """
m2 = """Section II - Statistics for USA (state-counties) """
m3 = """Section III - Statistics for other geographical regions (continent-countries)"""
m4 = """Section IV - Summary of maximum positive cases and maximum deaths around the globe"""
m5 = """Section V - Graphs(a: Positive cases and Deaths for all 5 continents, b: Variation in total positive cases for all 5 continents over time)"""
m6 = """ Section VI - Growth rate for each country in all 5 continents"""

message1 = """Section I."""
message2 = """Section II. First select state. Then select county."""
message3 = """Section III. First select continent. Then select country."""
message4 = """Section IV."""
message5 = """Section Va. Select the continent. Select Positive cases/Deaths."""
message6 = """Section Vb. Time series for total positive cases in all 5 continents."""
message7 = """Section VI. First seelct continent. Then select country."""
message8 = """ Note: Negative growth rate indicates decline in positive cases over time."""

#PREPARING` DATA REQUIRED FOR CALL-OUT OPERATIONS
all_options = {'America': listA4, 'Europe': listA2, 'Asia': listA3, 'Africa': listA1, 'Oceania': listA5}
all_options1 = {lblstate[v]:fl[v] for v in range(len(lblstate))}
all_options2 = {'America': aml, 'Europe': el, 'Asia': asl, 'Africa': afl, 'Oceania': ocl}

#DEVELOPING DASHBOARD CONTENT
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H6(children='Statistics and Data Visualization for COVID-19',style={'textAlign': 'center','color': colors['text']}),
    html.Div(children = introduction, style={'textAlign': 'left','color': colors['text'],'fontSize': 12,'margin-top':'15px'}),
    html.Div(children = m1, style={'textAlign': 'left','color': colors['text'],'fontSize': 12,'margin-top':'15px'}),
    html.Div(children = m2, style={'textAlign': 'left','color': colors['text'],'fontSize': 12,'margin-top':'15px'}),
    html.Div(children = m3, style={'textAlign': 'left','color': colors['text'],'fontSize': 12,'margin-top':'15px'}),
    html.Div(children = m4, style={'textAlign': 'left','color': colors['text'],'fontSize': 12,'margin-top':'15px'}),
    html.Div(children = m5, style={'textAlign': 'left','color': colors['text'],'fontSize': 12,'margin-top':'15px'}),
    html.Div(children = m6, style={'textAlign': 'left','color': colors['text'],'fontSize': 12,'margin-top':'15px'}),
    
    html.Div(children = message1, style={'textAlign': 'left','color': '#FFFD35','fontSize': 12,'margin-top':'15px'}),
    dash_table.DataTable(id = 'table',columns = [{'name':i,'id':i} for i in table_1.columns],data = table_1.to_dict('records'),fixed_rows={'headers':True},style_table={'height':200},
                         style_cell_conditional = [{'if': {'column_id': 'Continent'},'width': '75px'},
                                                   {'if': {'column_id': 'Country'},'width': '75px'},
                                                   {'if': {'column_id': 'State'},'width': '85px'},
                                                   {'if': {'column_id': 'Min positive cases'},'width': '80px'},
                                                   {'if': {'column_id': 'Max positive cases'},'width': '80px'},
                                                   {'if': {'column_id': 'Avg positive cases'},'width': '80px'},
                                                   {'if': {'column_id': 'Min deaths'},'width': '65px'},
                                                   {'if': {'column_id': 'Max deaths'},'width': '65px'},
                                                   {'if': {'column_id': 'Avg deaths'},'width': '65px'}],
                         style_cell = {'textAlign':'center','height':'30px','backgroundColor':'#DBE4F5','color':'#000000','fontWeight':'bold','border': '1px solid grey'},
                         style_header = {'fontWeight':'bold','color':'#000000','border': '1px solid grey'}),
    html.Div(children = message2, style={'textAlign': 'left','color': '#FFFD35','fontSize': 12,'margin-top':'15px'}),
    
    html.Div([dcc.Dropdown(id = 'State-dropdown',options=[{'label':k,'value':k} for k in all_options1.keys()],value = 'Alabama',style = {'width':'50%','fontSize':12}),
              dcc.Dropdown(id = 'County-dropdown',style = {'width':'50%','fontSize':12,'margin-top':'15px'}),
              html.Div(id = 'ftable',style = {'margin-top':'30px'})]),
    
    html.Div(children = message3, style={'textAlign': 'left','color': '#FFFD35','fontSize': 12,'margin-top':'15px'}),
    
    html.Div([dcc.Dropdown(id = 'Continent-dropdown',options=[{'label':k,'value':k} for k in all_options.keys()],value = 'America',style = {'width':'50%','fontSize':12}),
              dcc.Dropdown(id = 'Countries-dropdown',style = {'width':'50%','fontSize':12,'margin-top':'15px'}),
              html.Div(id = 'final_table',style = {'margin-top':'30px'})]),
    
    html.Div(children = message4, style={'textAlign': 'left','color': '#FFFD35','fontSize': 12,'margin-top':'15px'}),
    
    dash_table.DataTable(id = 'table2',columns = [{'name':i,'id':i} for i in table_4.columns],data = table_4.to_dict('records'),fixed_rows={'headers':True},
                         style_cell = {'textAlign':'center','height':'30px','backgroundColor':'#DBE4F5','color':'#000000','fontWeight':'bold','border': '1px solid grey'},
                         style_header = {'fontWeight':'bold','color':'#000000'}),
    
    html.Div(children = message5, style={'textAlign': 'left','color': '#FFFD35','fontSize': 12,'margin-top':'15px'}),
    
    html.Div([dcc.Dropdown(id = 'dropdown',options = [{'label':'America','value':'America'},
                                                      {'label':'Africa','value':'Africa'},
                                                      {'label':'Europe','value':'Europe'},
                                                      {'label':'Asia','value':'Asia'},
                                                      {'label':'Oceania','value':'Oceania'}],value = 'America',style = {'width':'50%','fontSize':12}),
              dcc.Dropdown(id = 'dropdown2',options = [{'label':'Positive cases','value':'Positive cases'},
                                                       {'label':'Deaths','value':'Deaths'}],value = 'Positive cases',style = {'width':'50%','fontSize':12,'margin-top':'15px'}),
              dcc.Graph(id = 'bar-graph')]),
    
    html.Div(children = message6, style={'textAlign': 'left','color': '#FFFD35','fontSize': 12,'margin-top':'15px'}),
    
    html.Div(dcc.Graph(figure = fig)),
    
    html.Div(children = message7, style={'textAlign': 'left','color': '#FFFD35','fontSize': 12,'margin-top':'15px'}),
    
    html.Div([dcc.Dropdown(id = 'continent-dd',options = [{'label':'America','value':'America'},
                                                      {'label':'Africa','value':'Africa'},
                                                      {'label':'Europe','value':'Europe'},
                                                      {'label':'Asia','value':'Asia'},
                                                      {'label':'Oceania','value':'Oceania'}],value = 'America',style = {'width':'50%','fontSize':12}),
              dcc.Dropdown(id = 'countries-dd',style = {'width':'50%','fontSize':12,'margin-top':'15px'}),
              html.Div(id = 'msg',style={'textAlign': 'left','color': colors['text'],'fontSize': 14,'fontWeight':'bold','margin-top':'15px'})]),
    
    html.Div(children = message8, style={'textAlign': 'left','color': '#8BFB37','fontSize': 12,'margin-top':'10px'})
              
])

#APP CALLBACK OPERATIONS (USER INTERACTION ENABLEMENT)
#for second set of dropdowns:
@app.callback(dash.dependencies.Output('Countries-dropdown','options'),
    [dash.dependencies.Input('Continent-dropdown','value')])
def set_countries_options(selected_continent):
    return [{'label': i, 'value': i} for i in all_options[selected_continent]]

@app.callback(dash.dependencies.Output('Countries-dropdown','value'),
    [dash.dependencies.Input('Countries-dropdown','options')])
def set_countries_values(available_options):
   return available_options[0]['value']

@app.callback(dash.dependencies.Output('final_table','children'),
    [dash.dependencies.Input('Continent-dropdown','value'),
     dash.dependencies.Input('Countries-dropdown','value')])
def set_display_children(selected_continent,selected_country):
    tab = table_2[(table_2['Continent'] == selected_continent) & (table_2['Country'] == selected_country)]
    return[dash_table.DataTable(id = 'table',columns = [{"name": i, "id": i} for i in tab.columns],data = tab.to_dict('records'),
                                style_cell_conditional = [{'if': {'column_id': 'Continent'},'width': '75px'},
                                                   {'if': {'column_id': 'Country'},'width': '75px'},
                                                   {'if': {'column_id': 'State'},'width': '110px'},
                                                   {'if': {'column_id': 'Min positive cases'},'width': '65px'},
                                                   {'if': {'column_id': 'Max positive cases'},'width': '65px'},
                                                   {'if': {'column_id': 'Avg positive cases'},'width': '65px'},
                                                   {'if': {'column_id': 'Min deaths'},'width': '65px'},
                                                   {'if': {'column_id': 'Max deaths'},'width': '65px'},
                                                   {'if': {'column_id': 'Avg deaths'},'width': '65px'}],
                         style_cell = {'textAlign':'center','height':'30px','backgroundColor':'#DBE4F5','color':'#000000','fontWeight':'bold','border': '1px solid grey'},
                         style_header = {'fontWeight':'bold','color':'#000000','border': '1px solid grey'})]

#for first set of dropdowns:
@app.callback(dash.dependencies.Output('County-dropdown','options'),
    [dash.dependencies.Input('State-dropdown','value')])
def set_County_options(selected_state):
    return [{'label': i, 'value': i} for i in all_options1[selected_state]]

@app.callback(dash.dependencies.Output('County-dropdown','value'),
    [dash.dependencies.Input('State-dropdown','options')])
def set_County_values(available_option):
    return available_option[0]['value']
    
@app.callback(dash.dependencies.Output('ftable','children'),
    [dash.dependencies.Input('State-dropdown','value'),
     dash.dependencies.Input('County-dropdown','value')])
def set_display_child(selected_state,selected_county):
    tab1 = table_3[(table_3['State'] == selected_state) & (table_3['County'] == selected_county)]
    return[dash_table.DataTable(id = 'table',columns = [{"name": i, "id": i} for i in tab1.columns],data = tab1.to_dict('records'),
                                style_cell_conditional = [{'if': {'column_id': 'Continent'},'width': '75px'},
                                                   {'if': {'column_id': 'Country'},'width': '75px'},
                                                   {'if': {'column_id': 'State'},'width': '75px'},
                                                   {'if': {'column_id': 'County'},'width': '75px'},
                                                   {'if': {'column_id': 'Min positive cases'},'width': '65px'},
                                                   {'if': {'column_id': 'Max positive cases'},'width': '65px'},
                                                   {'if': {'column_id': 'Avg positive cases'},'width': '65px'},
                                                   {'if': {'column_id': 'Min deaths'},'width': '65px'},
                                                   {'if': {'column_id': 'Max deaths'},'width': '65px'},
                                                   {'if': {'column_id': 'Avg deaths'},'width': '65px'}],
                         style_cell = {'textAlign':'center','height':'30px','backgroundColor':'#DBE4F5','color':'#000000','fontWeight':'bold','border': '1px solid grey'},
                         style_header = {'fontWeight':'bold','color':'#000000','border': '1px solid grey'})]

#for third set of dropdowns:
def create_figure(x1,y1,t,xlab):
    fig = go.Figure(go.Bar(x = x1,y = y1,orientation = 'h'))
    fig.update_layout(autosize = False, width = 1100, height = 480, font_color = '#FFFFFF', title_font_color = '#FFFFFF', title_font_size = 13,title_xref = 'paper',title_x = 0.7,margin = dict(l=20,r=20,b=30,t=30),paper_bgcolor = '#022C6A',title = {'text':f'Mean {xlab} in {t} from Dec 2019 to Oct 2020'},
                  xaxis_title = f'Mean {xlab}',yaxis_title = 'Countries')
    fig.update_xaxes(title_font=dict(size=12, color='#FFFFFF'))
    fig.update_yaxes(title_font=dict(size=12, color='#FFFFFF'))
    
    return fig

@app.callback(dash.dependencies.Output('bar-graph','figure'),
              [dash.dependencies.Input('dropdown','value'),
               dash.dependencies.Input('dropdown2','value')])
def update_figure(selected_value,sel_val):
    if selected_value == 'America' and sel_val == 'Positive cases':
        x1 = americalistp
        y1 = label2
        t = 'America'
        xlab = 'positive cases'
    if selected_value == 'Africa'and sel_val == 'Positive cases':
        x1 = africalistp
        y1 = label1
        t = 'Africa'
        xlab = 'positive cases'
    if selected_value == 'Asia'and sel_val == 'Positive cases':
        x1 = asialistp
        y1 = label3
        t = 'Asia'
        xlab = 'positive cases'
    if selected_value == 'Europe'and sel_val == 'Positive cases':
        x1 = europelistp
        y1 = label4
        t = 'Europe'
        xlab = 'positive cases'
    if selected_value == 'Oceania'and sel_val == 'Positive cases':
        x1 = oceanslistp
        y1 = label5
        t = 'Oceania'
        xlab = 'positive cases'
    if selected_value == 'America' and sel_val == 'Deaths':
        x1 = americalistd
        y1 = label2
        t = 'America'
        xlab = 'Deaths'
    if selected_value == 'Africa'and sel_val == 'Deaths':
        x1 = africalistd
        y1 = label1
        t = 'Africa'
        xlab = 'Deaths'
    if selected_value == 'Asia'and sel_val == 'Deaths':
        x1 = asialistd
        y1 = label3
        t = 'Asia'
        xlab = 'Deaths'
    if selected_value == 'Europe'and sel_val == 'Deaths':
        x1 = europelistd
        y1 = label4
        t = 'Europe'
        xlab = 'Deaths'
    if selected_value == 'Oceania'and sel_val == 'Deaths':
        x1 = oceanslistd
        y1 = label5
        t = 'Oceania'
        xlab = 'Deaths'
    fig = create_figure(x1,y1,t,xlab)
    return fig

#for fourth set of dropdowns:
@app.callback(dash.dependencies.Output('countries-dd','options'),
    [dash.dependencies.Input('continent-dd','value')])
def set_country_o(sel_conti):
    return [{'label': i, 'value': i} for i in all_options2[sel_conti]]

@app.callback(dash.dependencies.Output('countries-dd','value'),
    [dash.dependencies.Input('countries-dd','options')])
def set_country_v(av_op):
    return av_op[0]['value']

@app.callback(dash.dependencies.Output('msg','children'),
    [dash.dependencies.Input('countries-dd','value')])
def update_output(sel_count):
    val1 = str(table_6.loc[table_6['country'] == sel_count,'per_growth_rate'].iloc[0])
    return f'The growth rate for the above selection is {val1} %'

            
if __name__ == '__main__':
    app.run_server(debug=True)

#END OF PROGRAM
