from glob import glob
from lbhb.io import behavior
import os.path
import datetime as dt


from joblib import Memory
memory = Memory(cachedir='/tmp/lbhb_bburan_flask', verbose=0)


@memory.cache
def load_data():
    search_path = '/auto/data/daq/gerbil/behavior/170704.0 cohort'
    experiments = glob(os.path.join(search_path, '*appetitive_gonogo_food*'))
    return behavior.load_sam_behavior(experiments)


#data = load_data()
#data.to_pickle('tmp.pkl')
import pandas as pd
data = pd.read_pickle('tmp.pkl')
data['target_fc'] = data['target_fl'] * 0.5 + data['target_fh'] * 0.5
data['sens_level'] = data['target_level'] - data['masker_level']


DATE_FMT = '%Y %b %d'


from bokeh.layouts import column, row, widgetbox, layout
from bokeh.models import ColumnDataSource, Legend, RangeTool, Range1d
from bokeh.models.widgets import Select, MultiSelect, CheckboxButtonGroup
from bokeh.plotting import figure, curdoc
from bokeh.palettes import viridis

import numpy as np


def animal_changed():
    animal = animal_select.value
    update_animal_plots(animal)
    update_date_select(animal)


def update_animal_plots(animal):
    m_animal = data['animal'] == animal
    n_trials = data.loc[m_animal].groupby('session').size()
    animal_trials.data = {
        'session': n_trials.index.values,
        'trials': n_trials.values,
    }

    m = m_animal & (data['depth'] == 0.0)
    fa = data.loc[m].groupby('session')['yes'].mean()
    animal_fa.data = {
        'session': fa.index.values,
        'fa': fa.values,
    }


def update_date_select(animal):
    m = data['animal'] == animal
    dates = data.loc[m, 'date'].unique()
    dates = [d.strftime(DATE_FMT) for d in dates]
    date_select.options = dates


def create_plot():
    animal = animal_select.value
    dates = [dt.datetime.strptime(d, DATE_FMT).date() \
             for d in date_select.value]

    m = data['animal'] == animal
    if dates:
        m = m & data['date'].apply(lambda x: x in dates)

    group_labels = [groupby_select.labels[i] for i in groupby_select.active]
    group_keys = [groupby_options[l] for l in group_labels]

    psi = data.loc[m] \
        .groupby(group_keys + ['depth'])['yes'] \
        .agg(['size', 'sum', 'mean']) \
        .rename(columns={'size': 'n', 'sum': 'k', 'mean': 'p'}) \
        .unstack('depth')

    psi = psi.reindex(sorted(psi.columns), axis=1)

    plot = figure(title='psychometric function', tools="", x_range=(0, 1),
                  y_range=(0, 1))
    plot.xaxis.axis_label = 'AM depth (frac)'
    plot.yaxis.axis_label = 'Hit rate (frac)'

    colors = viridis(len(psi))
    n_ref = np.nanmean(psi['n'].values)
    legend_items = []

    for c, (level, row) in zip(colors, psi.iterrows()):
        row = row.dropna()
        d = row['p'].index.get_level_values('depth')
        p = row['p'].values
        n = row['n'].values / n_ref * 10
        l = plot.line(x=d, y=p, line_color=c)
        c = plot.circle(x=d, y=p, size=n, color=c)
        legend_items.append((str(level), [l, c]))

    legend = Legend(items=legend_items, location=(0, 0))
    plot.add_layout(legend, 'right')
    plot.legend.click_policy = 'hide'
    return plot


def update_plot():
    page_layout.children[3].children[1] = create_plot()


animals = data['animal'].unique().tolist()
animals.sort()
animal_select = Select(title='Animal', options=animals, value=animals[0])
animal_select.on_change('value', lambda attr, old, new: animal_changed())

date_select = MultiSelect(title='Dates', size=10)
date_select.on_change('value', lambda attr, old, new: update_plot())

groupby_options = {
    'masker level': 'masker_level',
    'sensation level': 'sens_level',
    'target center frequency': 'target_fc',
}

groupby_select = CheckboxButtonGroup(labels=list(groupby_options.keys()), active=[0])
groupby_select.on_change('active', lambda attr, old, new: update_plot())


animal_trials = ColumnDataSource()
animal_trials_plot = figure(title='number of trials', tools="", width=1000,
                             height=150)
animal_trials_plot.xaxis.axis_label = 'Session'
animal_trials_plot.yaxis.axis_label = 'Number of trials'
animal_trials_plot.line('session', 'trials', source=animal_trials)

animal_fa = ColumnDataSource()
animal_fa_plot = figure(title='False alarm rate', tools="", width=1000,
                        height=150, y_range=(0, 1))
animal_fa_plot.xaxis.axis_label = 'Session'
animal_fa_plot.yaxis.axis_label = 'FA rate'
animal_fa_plot.line('session', 'fa', source=animal_fa)


page_layout = layout(
    animal_select,
    animal_trials_plot,
    animal_fa_plot,
    [widgetbox(date_select, groupby_select), create_plot()],
)

# Run the updaters to initialize everything
animal_changed()
update_plot()

curdoc().add_root(page_layout)
