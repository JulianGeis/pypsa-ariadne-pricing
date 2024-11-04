# -*- coding: utf-8 -*-
import logging
import math
import os
import re
import sys
from functools import reduce

import pypsa
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from pypsa.descriptors import get_switchable_as_dense as as_dense


logger = logging.getLogger(__name__)

paths = [
    "workflow/submodules/pypsa-eur/scripts",
    "../submodules/pypsa-eur/scripts",
    "../submodules/pypsa-eur/",
]
for path in paths:
    sys.path.insert(0, os.path.abspath(path))

from _helpers import configure_logging, mute_print
from prepare_sector_network import prepare_costs


year_colors = ["dimgrey", "darkorange", "seagreen", "cadetblue", "hotpink", "darkviolet"]
markers =["v", "^", "<", ">", "1", "2", "3", "4", "*", "+", "d", "o", "|", "s", "P", "p", "h"]
date_format = "%Y-%m-%d %H:%M:%S"

carrier_renaming = {
    'urban central solid biomass CHP CC': 'biomass CHP CC',
    'urban central solid biomass CHP': 'biomass CHP',
    'urban central gas CHP': 'gas CHP',
    'urban central gas CHP CC': 'gas CHP CC',
    'urban central air heat pump': 'air heat pump',
    'urban central resistive heater': 'resistive heater',
    'urban central lignite CHP': 'lignite CHP'
}

carrier_renaming_reverse = {
    'biomass CHP CC': 'urban central solid biomass CHP CC',
    'biomass CHP' :'urban central solid biomass CHP' ,
    'gas CHP': 'urban central gas CHP' ,
    'gas CHP CC' : 'urban central gas CHP CC',
    'air heat pump' : 'urban central air heat pump',
    'resistive heater': 'urban central resistive heater',
    'lignite CHP': 'urban central lignite CHP'
}

def get_condense_sum(df, groups, groups_name, return_original=False):
    """
    return condensed df, that has been groupeb by condense groups
    Arguments:
        df: df you want to condense (carriers have to be in the columns)
        groups: group lables you want to condense on
        groups_name: name of the new grouped column
        return_original: boolean to specify if the original df should also be returned
    Returns:
        condensed df
    """
    result = df

    for group, name in zip(groups, groups_name):
        # check if carrier are in columns
        bool = [c in df.columns for c in group]
        # updated to carriers within group that are in columns
        group = list(compress(group, bool))

        result[name] = df[group].sum(axis=1)
        result.drop(group, axis=1, inplace=True)

    if return_original:
        return result, df

    return result

def timestep_before(timestep, n):
    if timestep == "2013-01-01 00:00:00" or timestep == "2019-01-01 00:00:00":
        return timestep
    dt = datetime.strptime(timestep, date_format)
    timestep_b = (dt - pd.Timedelta(n.snapshot_weightings.generators.iloc[0], unit="h")).strftime(date_format)
    return timestep_b


def supply_price_link(n, link, timestep, bus, co2_add_on=False):  
    cols = [col for col in n.links.columns if col.startswith("bus") and n.links.loc[link, col] != ""]
    loc_buses = n.links.loc[link, cols].tolist()
    j = loc_buses.index(bus)
    efficiency_link = 1 if j == 0 else n.links.loc[link, f"efficiency{j}" if j > 1 else 'efficiency']
    supply_price = 0

    for i, loc_bus in enumerate(loc_buses):
        efficiency_i = 1 if i == 0 else n.links.loc[link, f"efficiency{i}" if i > 1 else 'efficiency']

        if loc_bus == bus:
            # Cost for the specified bus
            price = n.buses_t.marginal_price.loc[timestep, n.links.loc[link, "bus0"]]
            mc = n.links.loc[link, 'marginal_cost']
            cost = (price + mc) / efficiency_link
            supply_price += cost
        elif (i > 0):
            # Revenue / costs at other buses
            price = price = n.buses_t.marginal_price.loc[timestep, loc_bus]
            if (n.buses.loc[loc_bus, "carrier"] in ["co2", "co2 stored"]) & co2_add_on:
                price += n.global_constraints.loc["co2_limit_upstream-DE"].mu
            rev = price * efficiency_i / efficiency_link         
            supply_price -= rev
    
    return supply_price

def demand_price_link(n, link, timestep, bus, co2_add_on=False, print_steps=False):  
    cols = [col for col in n.links.columns if col.startswith("bus") and n.links.loc[link, col] != ""]
    loc_buses = n.links.loc[link, cols].tolist()
    demand_price = 0
    if bus != n.links.loc[link, "bus0"]:
        for i, loc_bus in enumerate(loc_buses):
                if bus != loc_bus:
                    # Revenue at other buses (other from bus)
                    efficiency_i = -1 if i == 0 else as_dense(n, "Link", f"efficiency{i}" if i > 1 else 'efficiency').loc[timestep, link] # watch out for time dependent efficiencies of heat pumps: 
                    price = n.buses_t.marginal_price.loc[timestep, loc_bus]
                    if (n.buses.loc[loc_bus, "carrier"] in ["co2", "co2 stored"]) & co2_add_on:
                        price += n.global_constraints.loc["co2_limit_upstream-DE"].mu
                    rev = price * efficiency_i
                    if print_steps:
                        print(f"Revenue at bus {loc_bus}: {rev} from price {price} * efficiency {efficiency_i}")
                    demand_price += rev

    for i, loc_bus in enumerate(loc_buses):
        if i > 0:
                # Revenue at other buses (other from bus0)
                efficiency_i = 1 if i == 0 else as_dense(n, "Link", f"efficiency{i}" if i > 1 else 'efficiency').loc[timestep, link] # watch out for time dependent efficiencies of heat pumps: 
                price = n.buses_t.marginal_price.loc[timestep, loc_bus]
                if (n.buses.loc[loc_bus, "carrier"] in ["co2", "co2 stored"]) & co2_add_on:
                    price += n.global_constraints.loc["co2_limit_upstream-DE"].mu
                rev = price * efficiency_i
                if print_steps:
                    print(f"Revenue at bus {loc_bus}: {rev} from price {price} * efficiency {efficiency_i}")
                demand_price += rev
     
    demand_price -= n.links.loc[link, 'marginal_cost']
    if print_steps:
        print(f"Marginal cost of {link}: {n.links.loc[link, 'marginal_cost']}")
    
    return demand_price


def get_supply_demand(n, buses, timestep, co2_add_on=False):
    # Initialize DataFrames with the required columns
    supply = pd.DataFrame(columns=["mc", "capex_add_on", "p_nom_opt", "p", "volume_bid", "mc_final", "carrier"])
    demand = pd.DataFrame(columns=["bidding_price", "p", "p_nom_opt", "volume_demand", "carrier"])
    
    # Supply Calculation
    for gen in n.generators[n.generators.bus.isin(buses)].index:
        mc = n.generators.loc[gen].marginal_cost
        capex_add_on = -n.generators_t.mu_upper.loc[timestep, gen]
        p_nom_opt = n.generators.loc[gen].p_nom_opt
        p = n.generators_t.p.loc[timestep, gen]
        p_max_pu = n.generators_t.p_max_pu.loc[timestep, gen] if gen in n.generators_t.p_max_pu.columns else n.generators.loc[gen].p_max_pu
        volume_bid = p_nom_opt * p_max_pu
        mc_final = mc 
        carrier = n.generators.loc[gen].carrier
        
        supply.loc[gen] = [mc, capex_add_on, p_nom_opt, p, volume_bid, mc_final, carrier]

    for su in n.storage_units[n.storage_units.bus.isin(buses)].index:
        mc = n.storage_units.loc[su].marginal_cost + n.storage_units_t.mu_energy_balance.loc[timestep, su]  * 1/n.storage_units.efficiency_dispatch.loc[su]
        capex_add_on = -n.storage_units_t.mu_upper.loc[timestep, su]
        p_nom_opt = n.storage_units.loc[su].p_nom_opt
        p = n.storage_units_t.p_dispatch.loc[timestep, su]
        p_max_pu = n.storage_units.p_max_pu.loc[su]
        volume_bid = min(p_nom_opt * p_max_pu, n.storage_units_t.state_of_charge.loc[timestep_before(timestep, n), su]) #  * n.storage_units.efficiency_dispatch.loc[su] is alread in p_nom_opt integrated
        mc_final = mc 
        carrier = n.storage_units.loc[su].carrier
        
        supply.loc[su] = [mc, capex_add_on, p_nom_opt, p, volume_bid, mc_final, carrier]

    for st in n.stores[n.stores.bus.isin(buses)].index:
        mc = n.stores.loc[st].marginal_cost + n.stores_t.mu_energy_balance.loc[timestep, st]
        capex_add_on = -n.stores_t.mu_upper.loc[timestep, st]
        p_nom_opt = n.stores.loc[st].e_nom_opt
        p = n.stores_t.p.loc[timestep, st]
        volume_bid = min(n.stores.e_max_pu.loc[st] * n.stores.e_nom_opt.loc[st], n.stores_t.e.loc[timestep, st])
        mc_final = mc 
        carrier = n.stores.loc[st].carrier
        
        supply.loc[st] = [mc, capex_add_on, p_nom_opt, p, volume_bid, mc_final, carrier]

    loc_buses = ["bus" + str(i) for i in np.arange(0, 5)]
    for link in n.links.index:
        for bus in buses:
            # exclude if the bus is bus0 as there is no supply only demand for bus0
            if bus in n.links.loc[link, loc_buses[1:]].tolist():
                i = n.links.loc[link, loc_buses].tolist().index(bus)
                efficiency = 1 if i == 0 else (n.links.loc[link, 'efficiency'] if i == 1 else n.links.loc[link, f"efficiency{i}"])
                bus_from = n.links.loc[link, "bus0"]
                # assumption that the marginal costs only apply to bus 1 (main output) 
                mc =  n.links.loc[link].marginal_cost # - (n.links_t.mu_upper.loc[timestep, link] + n.links.loc[link].marginal_cost if i == 1 else 0)
                capex_add_on = -n.links_t.mu_upper.loc[timestep, link]
                p_nom_opt = n.links.loc[link].p_nom_opt * efficiency
                p = -n.links_t[f"p{i}"].loc[timestep, link]
                p_max_pu = n.links_t.p_max_pu.loc[timestep, link] if link in n.links_t.p_max_pu.columns else n.links.loc[link].p_max_pu
                volume_bid = p_nom_opt * p_max_pu # efficiency is already included in p_nom_opt
               # cost from buying one unit of input energy (bus_from); 
                mc_final = supply_price_link(n, link, timestep, bus, co2_add_on) #or: mc + supply_price_link_old(n, link, timestep, bus)
                carrier = n.links.loc[link].carrier
                
                supply.loc[link] = [mc, capex_add_on, p_nom_opt, p, volume_bid, mc_final, carrier]

    loc_buses = ["bus0", "bus1"]
    for line in n.lines.index:
        for bus in buses:
            if bus in n.lines.loc[line, loc_buses].tolist():
                i = n.lines.loc[line, loc_buses].tolist().index(bus)
                bus_from = n.lines.loc[line, "bus0"] if bus == n.lines.loc[line, "bus0"] else n.lines.loc[line, "bus1"]
                mc = n.buses_t.marginal_price.loc[timestep, bus_from]
                capex_add_on = -n.lines_t.mu_upper.loc[timestep, line]
                p_nom_opt = n.lines.loc[line].s_nom_opt
                p = -n.lines_t[f"p{i}"].loc[timestep, line]
                volume_bid = n.lines.s_max_pu.loc[line] * n.lines.s_nom_opt.loc[line]
                mc_final = mc 
                carrier = n.lines.loc[line].carrier
                
                supply.loc[line] = [mc, capex_add_on, p_nom_opt, p, volume_bid, mc_final, carrier]

    # Demand Calculation
    for load in n.loads[n.loads.bus.isin(buses)].index:
        bidding_price = np.inf
        p = n.loads_t.p.loc[timestep, load]
        p_nom_opt = p
        volume_demand = p
        carrier = n.loads.loc[load].carrier
        
        demand.loc[load] = [bidding_price, p, p_nom_opt, volume_demand, carrier]

    for su in n.storage_units[n.storage_units.bus.isin(buses)].index:
        bidding_price = -n.storage_units_t.mu_upper.loc[timestep, su] + n.storage_units_t.mu_energy_balance.loc[timestep, su] * n.storage_units.efficiency_store.loc[su] 
        p = n.storage_units_t.p_store.loc[timestep, su]
        p_nom_opt = n.storage_units.loc[su].p_nom_opt
        storage_space =  n.storage_units.loc[su].max_hours * p_nom_opt - n.storage_units_t.state_of_charge.loc[timestep_before(timestep, n), su]
        volume_demand = min(abs(p_nom_opt * n.storage_units.p_min_pu.loc[su]), storage_space) # this value is negative  * n.storage_units.efficiency_store.loc[su] already integrated in p_nom_opt
        carrier = n.storage_units.loc[su].carrier
        
        demand.loc[su] = [bidding_price, p, p_nom_opt, volume_demand, carrier]

    for st in n.stores[n.stores.bus.isin(buses)].index:
        bidding_price = -n.stores_t.mu_upper.loc[timestep, st] + n.stores_t.mu_energy_balance.loc[timestep, st]
        p = -n.stores_t.p.loc[timestep, st]
        p_nom_opt = n.stores.loc[st].e_nom_opt
        volume_demand = abs(p_nom_opt * n.stores.e_nom_min.loc[st]) # this value is negative
        carrier = n.stores.loc[st].carrier
        
        demand.loc[st] = [bidding_price, p, p_nom_opt, volume_demand, carrier]

    # only working for demand bid at bus0 ( are there links with several inputs and how to handle them? Yes, methanolisation)
    loc_buses = ["bus" + str(i) for i in np.arange(0, 5)]
    for link in n.links.index:
        for bus in buses:
            if (bus in n.links.loc[link, loc_buses].tolist()) & (bus == n.links.loc[link, "bus0"]):
                i = n.links.loc[link, loc_buses].tolist().index(bus)
                efficiency = 1 if i == 0 else (n.links.loc[link, 'efficiency'] if i == 1 else n.links.loc[link, f"efficiency{i}"])
                bidding_price = demand_price_link(n, link, timestep, bus, co2_add_on)
                p = n.links_t[f"p{i}"].loc[timestep, link]
                p_nom_opt = n.links.loc[link].p_nom_opt #* efficiency
                p_min_pu = n.links_t.p_min_pu.loc[timestep, link] if link in n.links_t.p_min_pu.columns else n.links.loc[link].p_min_pu
                if n.links.loc[link].carrier == "BEV charger":
                    volume_demand = min(abs(p_nom_opt),n.loads_t.p.loc[timestep, "DE0 0 land transport EV"]) 
                if n.links.loc[link].carrier == "urban decentral resistive heater":
                    volume_demand = min(abs(p_nom_opt),n.loads_t.p.loc[timestep, "DE0 0 urban decentral heat"]/efficiency) 
                if n.links.loc[link].carrier == "rural resistive heater":
                    volume_demand = min(abs(p_nom_opt),n.loads_t.p.loc[timestep, "DE0 0 rural heat"]/efficiency)
                else:
                    volume_demand = abs(p_nom_opt)
                carrier = n.links.loc[link].carrier
                demand.loc[link] = [bidding_price, p, p_nom_opt, volume_demand, carrier]
                
            elif (bus in n.links.loc[link, loc_buses].tolist()):
                # if there is another bus, the efficiency has to be negative
                i = n.links.loc[link, loc_buses].tolist().index(bus)
                efficiency = 1 if i == 0 else (n.links.loc[link, 'efficiency'] if i == 1 else n.links.loc[link, f"efficiency{i}"])
                if efficiency > 0:
                    continue
                bidding_price = demand_price_link(n, link, timestep, bus, co2_add_on)
                p = n.links_t[f"p{i}"].loc[timestep, link]
                p_nom_opt = abs(n.links.loc[link].p_nom_opt * efficiency)
                volume_demand = p_nom_opt
                carrier = n.links.loc[link].carrier
                demand.loc[link] = [bidding_price, p, p_nom_opt, volume_demand, carrier]

            

    loc_buses = ["bus0", "bus1"]
    for line in n.lines.index:
        for bus in buses:
            if bus in n.lines.loc[line, loc_buses].tolist():
                i = n.lines.loc[line, loc_buses].tolist().index(bus)
                bidding_price = n.buses_t.marginal_price.loc[timestep, bus]
                p = n.lines_t[f"p{i}"].loc[timestep, line]
                p_nom_opt = n.lines.loc[line].s_nom_opt
                volume_demand = p_nom_opt
                carrier = n.lines.loc[line].carrier
                
                demand.loc[line] = [bidding_price, p, p_nom_opt, volume_demand, carrier]

    return supply, demand

def plot_supply_demand(n, 
                    supply, 
                    demand, 
                    buses, 
                    timestep,
                    tech_colors, 
                    ylim=None, 
                    p="p_nom_opt", 
                    d="p_nom_opt", 
                    mc="mc", 
                    savepath=None,
                    only_carriers=False, 
                    whole_system=False, 
                    demand_plot=True, 
                    demand_text=True, 
                    compress_demand=False, 
                    year=9999):
    # Filter out technologies with negative supply
    drop_c = ["electricity distribution grid", "load", "BEV charger"]
    supply = supply[(supply[p] >= 1) & ~(supply.carrier.isin(drop_c))]

    # Convert to GW
    supply.loc[:, p] = supply[p] / 1e3
    
    if whole_system:
        supply = supply[~supply.carrier.isin(["AC", "DC"])]
        demand = demand[~demand.carrier.isin(["AC", "DC"])]

    # Sort the supply DataFrame by the marginal cost column
    supply = supply.sort_values(by=mc)

    # Process the demand data
    demand = demand[demand[d] > 1]  # Drop all technologies with demand less than 1e-1 MW
    demand[d] = demand[d] / 1e3  # Convert to GW
    demand.bidding_price = demand.bidding_price.clip(lower=0) # Clip negative bidding prices to 0 
    # group demand technologies together, where the bidding price differene is lower than a threshold
    if compress_demand:
        demand = get_compressed_demand(demand, th=0.1)
        only_carriers = True # no more index available as data is grouped
    demand = demand.sort_values(by='bidding_price', ascending=False)
    
    # Add start and end of demand curve for plotting
    start_row = pd.DataFrame([{"bidding_price": np.inf, "p": 0, "p_nom_opt": 0, "volume_demand": 0, "carrier": "start"}], index=["start"])
    end_row = pd.DataFrame([{"bidding_price": 0, "p": 0, "p_nom_opt": 0,"volume_demand": 0, "carrier": "end"}], index=["end"])
    demand = pd.concat([start_row, demand, end_row])
    
    # Replace infinite bidding price with 10% more than the maximum finite bidding price
    max_finite_bidding_price = demand.bidding_price[demand.bidding_price.apply(np.isfinite)].max()
    demand["bidding_price"] = demand["bidding_price"].replace(np.inf, max_finite_bidding_price * 1.1)
    
    # Cumulative sum of demand
    demand[d] = demand[d].cumsum()

    if compress_demand:
        demand.reset_index(drop=True, inplace=True)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    x_position = 0

    supply_labels = set()  # Set to track labels to avoid duplicates

    # Plot supply bars
    for index, row in supply.iterrows():
        height = row[mc]
        width = row[p]
        carrier_label = str(row['carrier']) if only_carriers else index
        ax.add_patch(plt.Rectangle((x_position, 0), width, height, color=tech_colors[str(row['carrier'])], 
                                   alpha=0.5, label=carrier_label if carrier_label not in supply_labels else "_nolegend_"))
        supply_labels.add(carrier_label)
        x_position += width

    ax.set_xlim(0, max(x_position, demand[d].max()) * 1.11)
    ax.set_ylim(0, max(demand.bidding_price.max(), supply[mc].max()) * 1.11)
    if ylim is not None:
        ax.set_ylim(0, ylim)

    plt.xlabel('bid and ask volume [GW]')
    plt.ylabel('bid and ask price [€/MWh]')
    plt.title(f'Market clearing for electricity at {timestep}')

    demand_legend = {}  # Dictionary to store the number-label pairs for the legend

    if demand_plot:
        # Plot demand curve
        for i in range(len(demand) - 1):
            x1 = demand.iloc[i][d]
            x2 = demand.iloc[i + 1][d]
            y1 = demand.iloc[i + 1]['bidding_price']
            y2 = demand.iloc[i + 1]['bidding_price']
            ax.plot([x1, x2], [y1, y2], color='black', linestyle='--')
            if (demand_text & (i < len(demand) - 2)):
                number_label = str(i + 1)  # Number to display in the plot
                ax.text((x1 + x2) / 2, ((y1 + y2) / 2)+ 7, number_label, fontsize=12, ha='center', rotation=0)
                demand_legend[number_label] = (demand.index[i + 2] if not only_carriers else demand.carrier[i + 1])
            ax.plot([x1, x1], [demand.iloc[i]['bidding_price'], y2], color='grey', linestyle='--')


    # Create supply legend handles and labels
    supply_handles, supply_labels = ax.get_legend_handles_labels()
    supply_handles_labels = dict(zip(supply_labels, supply_handles))
    supply_handles_labels.pop("_nolegend_", None)  # Remove the dummy label used to avoid duplicates

    # Create demand legend handles and labels
    demand_legend_handles = [plt.Line2D([0], [0], color='white', label=f"{num}: {label}") for num, label in demand_legend.items()]

    # Plot market clearing point if there is only one bus
    if len(buses) == 1:
        ax.plot(n.statistics.supply(bus_carrier="AC", aggregate_time=False)[timestep].sum()/1e3, n.buses_t.marginal_price.loc[timestep, buses], 
                 marker='x', markersize=7, color="red", label='market clearing')


    legend1 = plt.legend(handles=supply_handles_labels.values(), labels=supply_handles_labels.keys(), title="Supply",
                        loc='upper left', bbox_to_anchor=(1, 1.02))
    ax.add_artist(legend1)

    legend2 = plt.legend(handles=demand_legend_handles, title="Demand", loc='upper left', bbox_to_anchor=(0, -0.12), ncol=2)
    ax.add_artist(legend2)
    # plt.tight_layout()

    plt.grid(True)
    fig.savefig(savepath, bbox_extra_artists=(legend1,legend2), bbox_inches='tight')

def get_compressed_demand(demand, th):
    demand.loc[:, "carrier"] = demand["carrier"].replace({
        "electricity": "electricity load", 
        "industry electricity": "electricity load", 
        "agriculture electricity": "electricity load"
    })
    df = demand.sort_values(by=['carrier', 'bidding_price']).reset_index(drop=True)
    group = (df['bidding_price'].diff().abs() > th).cumsum()
    df['group'] = group

    grouped_df = df.groupby(['carrier', 'group']).agg({
        'bidding_price': 'mean',  
        'p': 'sum',               
        'p_nom_opt': 'sum',       
        'volume_demand': 'sum',    
        'carrier': 'first',  
    }).reset_index(drop=True)

    return grouped_df

    # side by side merit order plot
def plot_supply_demand_s(n, supply, demand, buses, timestep, ylim=None, p="p_nom_opt", d="p_nom_opt", mc="mc", 
                       only_carriers=False, whole_system=False, demand_plot=True, demand_text=True, 
                       compress_demand=False, ax=None, year=9999):
    # Filter out technologies with negative supply
    drop_c = ["electricity distribution grid", "load", "BEV charger"]
    supply = supply[(supply[p] >= 5) & ~(supply.carrier.isin(drop_c))]
    supply[p] = supply[p] / 1e3 # Convert to GW
    
    if whole_system:
        supply = supply[~supply.carrier.isin(["AC", "DC"])]
        demand = demand[~demand.carrier.isin(["AC", "DC"])]
    
    supply = supply.sort_values(by=mc)
    demand = demand[demand[d] > 1]
    demand[d] = demand[d] / 1e3
    demand.bidding_price = demand.bidding_price.clip(lower=0) # Clip negative bidding prices to 0
    
    if compress_demand:
        demand = get_compressed_demand(demand, th=0.1)
        only_carriers = True

    demand = demand.sort_values(by='bidding_price', ascending=False)
    
    start_row = pd.DataFrame([{"bidding_price": np.inf, "p": 0, "p_nom_opt": 0, "volume_demand": 0, "carrier": "start"}], index=["start"])
    end_row = pd.DataFrame([{"bidding_price": 0, "p": 0, "p_nom_opt": 0, "volume_demand": 0, "carrier": "end"}], index=["end"])
    demand = pd.concat([start_row, demand, end_row])
    
    max_finite_bidding_price = demand.bidding_price[demand.bidding_price.apply(np.isfinite)].max()
    demand["bidding_price"] = demand["bidding_price"].replace(np.inf, max_finite_bidding_price * 1.1)
    demand[d] = demand[d].cumsum()

    if compress_demand:
        demand.reset_index(drop=True, inplace=True)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    x_position = 0
    supply_labels = set()
    
    for index, row in supply.iterrows():
        height = row[mc]
        width = row[p]
        carrier_label = str(row['carrier']) if only_carriers else index
        ax.add_patch(plt.Rectangle((x_position, 0), width, height, color=tech_colors[str(row['carrier'])], 
                                   alpha=0.5, label=carrier_label if carrier_label not in supply_labels else "_nolegend_"))
        supply_labels.add(carrier_label)
        x_position += width

    ax.set_xlim(0, max(x_position, demand[d].max()) * 1.11)
    ax.set_ylim(0, max(demand.bidding_price.max(), supply[mc].max()) * 1.11)
    if ylim is not None:
        ax.set_ylim(0, ylim)
        
    if len(buses) == 1:
        ax.plot(n.statistics.supply(bus_carrier="AC", aggregate_time=False)[timestep].sum(), 
                n.buses_t.marginal_price.loc[timestep, buses], marker='x', markersize=7, color="red", label='market clearing')
        
    ax.set_xlabel('bid and ask volume [GW]')
    ax.set_ylabel('bid and ask price [€/MWh]')
    ax.set_title(f'Market clearing for electricity at {timestep} in year {year}')
    
    demand_legend = {}

    if demand_plot:
        for i in range(len(demand) - 1):
            x1 = demand.iloc[i][d]
            x2 = demand.iloc[i + 1][d]
            y1 = demand.iloc[i + 1]['bidding_price']
            y2 = demand.iloc[i + 1]['bidding_price']
            ax.plot([x1, x2], [y1, y2], color='black', linestyle='--')
            if (demand_text & (i < len(demand) - 2)):
                number_label = str(i + 1)
                ax.text((x1 + x2) / 2, ((y1 + y2) / 2) + 7, number_label, fontsize=12, ha='center', rotation=0)
                demand_legend[number_label] = (demand.index[i + 2] if not only_carriers else demand.carrier[i + 1])
            ax.plot([x1, x1], [demand.iloc[i]['bidding_price'], y2], color='grey', linestyle='--')

    supply_handles, supply_labels = ax.get_legend_handles_labels()
    supply_handles_labels = dict(zip(supply_labels, supply_handles))
    supply_handles_labels.pop("_nolegend_", None)

    demand_legend_handles = [plt.Line2D([0], [0], color='white', label=f"{num}: {label}") for num, label in demand_legend.items()]

    # Plot market clearing point if there is only one bus
    if len(buses) == 1:
        ax.plot(n.statistics.supply(bus_carrier="AC", aggregate_time=False)[timestep].sum()/1e3, n.buses_t.marginal_price.loc[timestep, buses], 
                 marker='x', markersize=7, color="red", label='market clearing')

    
    supply_handles, supply_labels = ax.get_legend_handles_labels()
    supply_handles_labels = dict(zip(supply_labels, supply_handles))
    supply_handles_labels.pop("_nolegend_", None)

    legend2 = ax.legend(handles=demand_legend_handles, title="Demand", loc='upper right', ncol=2, fancybox=True, framealpha=0.3)
    ax.add_artist(legend2)

    ax.grid(True)

    # Return the legend handles and labels
    return supply_handles_labels

def get_all_supply_prices(n, bus, period=None, carriers=None):

    index = n.snapshots if period is None else period

    # Initialize a dictionary to store results temporarily
    res_dict = {ts: {} for ts in index}

    for gen in n.generators.index[(n.generators.bus == bus) & ((n.generators.carrier.isin(carriers)) if carriers is not None else True)]:
        marginal_cost = n.generators.loc[gen].marginal_cost
        for ts in index:
            res_dict[ts][gen] = marginal_cost

    for su in n.storage_units.index[(n.storage_units.bus == bus) & ((n.storage_units.carrier.isin(carriers)) if carriers is not None else True)]:
        for ts in index:
            res_dict[ts][su] = (
                n.storage_units.loc[su].marginal_cost +
                n.storage_units_t.mu_energy_balance.loc[ts, su] *
                1 / n.storage_units.efficiency_dispatch.loc[su]
            )

    for st in n.stores.index[(n.stores.bus == bus) & ((n.stores.carrier.isin(carriers)) if carriers is not None else True)]:
        for ts in index:
            res_dict[ts][st] = (
                n.stores.loc[st].marginal_cost +
                n.stores_t.mu_energy_balance.loc[ts, st]
            )

    loc_buses = ["bus" + str(i) for i in np.arange(0, 5)]
    for link in n.links.index[(n.links.bus0 != bus) & (n.links[loc_buses].isin([bus]).any(axis=1)) & ((n.links.carrier.isin(carriers)) if carriers is not None else True)]:
        for ts in index:
            res_dict[ts][link] = supply_price_link(n, link, ts, bus)

    # Convert the dictionary to a DataFrame
    res = pd.DataFrame.from_dict(res_dict, orient='index')
    
    return res


def get_all_demand_prices(n, bus, period=None, carriers=None):

    index = n.snapshots if period is None else period

    # Initialize a dictionary to store results temporarily
    res_dict = {ts: {} for ts in index}

    for su in n.storage_units.index[(n.storage_units.bus == bus) & ((n.storage_units.carrier.isin(carriers))  if carriers is not None else True)]:
        for ts in index:
            res_dict[ts][su] = (-n.storage_units_t.mu_upper.loc[ts, su] + n.storage_units_t.mu_energy_balance.loc[ts, su])

    for st in n.stores.index[(n.stores.bus == bus) & ((n.stores.carrier.isin(carriers)) if carriers is not None else True)]:
        for ts in index:
            res_dict[ts][st] = (-n.stores_t.mu_upper.loc[ts, st] + n.stores_t.mu_energy_balance.loc[ts, st])

    loc_buses = ["bus" + str(i) for i in np.arange(0, 5)]
    for link in n.links.index[(n.links[loc_buses].isin([bus]).any(axis=1)) & ((n.links.carrier.isin(carriers)) if carriers is not None else True)]:
        for ts in index:
            res_dict[ts][link] = demand_price_link(n, link, ts, bus)

    # Convert the dictionary to a DataFrame
    res = pd.DataFrame.from_dict(res_dict, orient='index')
    
    return res

def price_setter(n, bus, timestep, co2_add_on=False, suppress_warnings=False):
    mp = n.buses_t.marginal_price.loc[timestep, bus]
    if not isinstance(bus, list):
        bus = [bus]
    supply, demand = get_supply_demand(n, bus, timestep, co2_add_on)

    # Filter where supply (p) is greater than 0.1 (using 1 omits the marginal generator at some times)
    th_p = 1e-3
    drop_c_s = ["electricity distribution grid", "load", "BEV charger"]
    supply = supply[(supply.p > th_p) & ~(supply.carrier.isin(drop_c_s))]
    demand = demand[demand.p > th_p]
    
    # Find supply items where 'mc_final' is equal to the marginal price
    supply_closest = supply[supply['mc_final'] == mp]
    demand_closest = demand[demand['bidding_price'] == mp]

    # If no exact match, find the closest supply item to the marginal price
    if supply_closest.empty:
        closest_index = (supply['mc_final'] - mp).abs().argsort()[:1]
        supply_closest = supply.iloc[closest_index]
    else:
        closest_index = (supply['mc_final'] - mp).abs().argsort()[:1]
        supply_closest = pd.concat([supply_closest, supply.iloc[closest_index]]).drop_duplicates()

    if demand_closest.empty:
        closest_index = (demand['bidding_price'] - mp).abs().argsort()[:1]
        demand_closest = demand.iloc[closest_index]
    else:
        closest_index = (demand['bidding_price'] - mp).abs().argsort()[:1]
        demand_closest = pd.concat([demand_closest, demand.iloc[closest_index]]).drop_duplicates()  

    sc = supply_closest.copy()   
    sc["bus"] = bus[0]
    sc["timestep"] = timestep
    sc["supply_price"] = supply.mc_final[sc.index]
    sc["marginal price @ bus"] = mp
    sc["sp - mp"] = sc.loc[:,"supply_price"] - sc.loc[:,"marginal price @ bus"]
    sc["capacity_usage"] = sc.loc[:,"p"] / sc.loc[:,"volume_bid"]

    # checks
    if not suppress_warnings:
        if abs(sc["sp - mp"].values[0]) > 1: # diff of marginal price at bus and supply price
            loggger.warning(f"Warning: Supply price differs from market clearing price by {sc['sp - mp'].values[0]}; supply_price: {supply.mc_final[sc.index].iloc[0]}, marginal_price @ bus: {mp} (timestep {timestep})")
        if sc["capacity_usage"].values[0] > 0.999: # check capacity usage
            logger.warning(f"Warning: Marginal generator uses full capacity {sc['capacity_usage'].values[0]} (timestep {timestep})")    
        p_s = supply[supply.p > th_p].sort_values(by="mc_final", ascending=True)[:supply_closest.index[0]].p.sum()
        p_s_true = n.statistics.supply(bus_carrier="AC", aggregate_time=False)[timestep].sum()
        if (p_s - p_s_true) > 10:
            logger.warning(f"Warning: Supply does not match the total supply {p_s} != {p_s_true} (timestep {timestep})")  

    dc = demand_closest.copy()   
    dc["bus"] = bus[0]
    dc["timestep"] = timestep
    dc["bidding_price"] = demand.bidding_price[dc.index]
    dc["marginal price @ bus"] = mp
    dc["bp - mp"] = dc.loc[:,"bidding_price"] - dc.loc[:,"marginal price @ bus"]
    dc["capacity_usage"] = dc.loc[:,"p"] / dc.loc[:,"volume_demand"]
    
    return sc, dc

if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "pricing_analysis",
            simpl="",
            clusters=27,
            opts="",
            ll="vopt",
            sector_opts="None",
            run="KN2045_Bal_v4",
        )
    
    # ensure output directory exist
    for dir in snakemake.output[1:]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    configure_logging(snakemake)
    config = snakemake.config
    planning_horizons = snakemake.params.planning_horizons
    nhours = int(snakemake.params.hours[:-1])
    nyears = nhours / 8760
    tech_colors = snakemake.params.plotting["tech_colors"]

    costs = list(
        map(
            lambda _costs: prepare_costs(
                _costs,
                snakemake.params.costs,
                nyears,
            ).multiply(
                1e-9
            ),  # in bn €
            snakemake.input.costs,
        )
    )
    # Load data
    _networks = [pypsa.Network(fn) for fn in snakemake.input.networks]
    modelyears = [fn[-7:-3] for fn in snakemake.input.networks]
     
    if snakemake.params.transmission_projects:   
        # Hack the transmission projects
        networks = [
            hack_transmission_projects(n.copy(), _networks[0], int(my), snakemake, costs)
            for n, my in zip(_networks, modelyears)
        ]
    else:
        networks = _networks

    # save as dict
    n_dict = {}
    years = np.arange(2020,2050,5)
    for i, year in enumerate(years):
        n_dict[year] = networks[i]

    # update tech_colors
    colors_update = networks[0].carriers.color.rename(networks[0].carriers.nice_name).to_dict()
    colors_update = {k: v for k, v in colors_update.items() if v != ""}
    tech_colors.update(colors_update)

    # carrier names manual
    tech_colors["urban central oil CHP"] = tech_colors["oil"]
    tech_colors["Solar"] = tech_colors["solar"]
    tech_colors["Electricity load"] = tech_colors["electricity"]
    tech_colors["Electricity trade"] = tech_colors["AC"]
    tech_colors["Offshore Wind"] = tech_colors["offwind-ac"]
    tech_colors["urban decentral heat"] = tech_colors["urban central heat"]
    tech_colors["urban decentral biomass boiler"] = tech_colors["biomass boiler"]
    tech_colors["rural biomass boiler"] = tech_colors["biomass boiler"]
    tech_colors["urban decentral oil boiler"] = tech_colors["oil boiler"]
    tech_colors["rural oil boiler"] = tech_colors["oil boiler"]
    tech_colors["rural ground heat pump"] = tech_colors["ground heat pump"]
    tech_colors['gas CHP'] = 'darkorange'
    tech_colors['urban decentral resistive heater'] = 'indianred' 
    tech_colors['urban decentral air heat pump'] = 'salmon'
    tech_colors['rural resistive heater'] = "indianred"
    tech_colors['rural air heat pump'] = "salmon"
    
    # calc price setter info
    networks = n_dict
    results_s = {}
    results_d = {}

    for year in planning_horizons:
        n = networks[year]
        res_s = pd.DataFrame()
        res_d = pd.DataFrame()
        for bus in n.buses.query("carrier == 'AC'").index:
            for snapshot in n.buses_t.p.index:
                s, d = price_setter(n, bus, str(snapshot), suppress_warnings=False)
                res_s = pd.concat([res_s, s])
                res_d = pd.concat([res_d, d])
            results_s[year]  = res_s
            results_d[year]  = res_d
    
    # save as pickle
    path_s = snakemake.output.pricing + '/res_1cl_3H_s.pkl'
    path_d = snakemake.output.pricing + '/res_1cl_3H_d.pkl'
    with open(path_s, 'wb') as file:
        pickle.dump(results_s, file)
    with open(path_d, 'wb') as file:
        pickle.dump(results_d, file)

    # obtain all supply (bid) and demand (ask) prices and then divide in if they are accepted or not
    bus = "DE0 0"
    bid = {}
    ask = {}

    for year in planning_horizons:
        n = networks[year]
        bid[year] = get_all_supply_prices(n, bus)
        ask[year] = get_all_demand_prices(n, bus)

    # save as pickle
    path_s = snakemake.output.pricing + '/bid.pkl'
    path_d = snakemake.output.pricing + '/ask.pkl'
    with open(path_bid, 'wb') as file:
        pickle.dump(bid, file)
    with open(path_ask, 'wb') as file:
        pickle.dump(ask, file)

    # # debugging 
    # path = "/home/julian-geis/repos/01_pricing-paper/pricing_analysis/data/results/20241031-OneNode-DownstreamVsUpstream-NoDistGrid"
    # results_s = pickle.load(open(path + "/res_1cl_3H_s.pkl", "rb"))
    # results_d = pickle.load(open(path + "/res_1cl_3H_d.pkl", "rb"))
    # bid = pickle.load(open(path + "/bid.pkl", "rb"))
    # ask = pickle.load(open(path + "/ask.pkl", "rb"))

    # plotting - 3 cases
    for year in planning_horizons:

        ts = ["2019-01-11 15:00:00", "2019-08-05 18:00:00",  "2019-06-02 12:00:00"] 
        all_supply_handles_labels = {}
        num_subplots = 3
        fig, axes = plt.subplots(3, 1, figsize=(8, 3*6))
        axes = axes.flatten()

        # Plot each subplot and collect legend handles and labels
        for i in range(num_subplots):
            n = networks[year]
            buses = ["DE0 0"]
            timestep = ts[i]
            supply, demand = get_supply_demand(n, buses, timestep)
            supply_handles_labels = \
                plot_supply_demand_s(n, supply, demand, buses, timestep, p="volume_bid", d="volume_demand", mc="mc_final",
                                only_carriers=True, demand_text=True, compress_demand=True, ax=axes[i], year=year)

            # Merge the current subplot's supply legend with the global collection
            all_supply_handles_labels.update(supply_handles_labels)
        del all_supply_handles_labels["market clearing"]    

        # Create the combined supply legend
        fig.legend(handles=all_supply_handles_labels.values(),
                labels=all_supply_handles_labels.keys(),
                title="Supply", loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))

        plt.tight_layout(pad=2)
        plt.savefig(f"{snakemake.output.merit_order_3cases}/{year}.png", bbox_inches='tight')
        plt.close(fig) 

    # plotting - all merit orders
    for year in planning_horizons:
        n = networks[year]
        # plot only every 5th timestep
        for timestep in n.snapshots[::5]:
            supply, demand = get_supply_demand(n, buses, str(timestep))
            plot_supply_demand(n, 
                    supply, 
                    demand, 
                    buses, 
                    str(timestep), 
                    tech_colors,
                    ylim=None, 
                    p="volume_bid", 
                    d="volume_demand", 
                    mc="mc_final",
                    savepath=f"{snakemake.output.merit_order_all}/{year}-{timestep}.png",
                    only_carriers=True, 
                    whole_system=False, 
                    demand_plot=True, 
                    demand_text=True, 
                    compress_demand=True, 
                    year=year)

    # plotting - price setter
    data = results_s

    for year in planning_horizons:
        df = data[year].copy()
        df.set_index("timestep", inplace=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        for carrier in df.carrier.unique():
            df["marginal price @ bus"][df.carrier == carrier].plot(style='.', label=carrier, color=tech_colors[carrier])
        plt.ylim(0,300)
        plt.title(f"Price setter for electricity in {year}")
        plt.legend(bbox_to_anchor=(1, 1))
        plt.savefig(f"{snakemake.output.price_setter}/{year}-price-setter.png", bbox_inches='tight')

    
    # plotting - price taker
    data = results_d

    for year in planning_horizons:
        df = data[year].copy()
        df = df[df.bidding_price.notna()]
        df.set_index("timestep", inplace=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        for carrier in df.carrier.unique():
            df["marginal price @ bus"][df.carrier == carrier].plot(style='.', label=carrier, color=tech_colors[carrier])
        plt.ylim(0,200)
        plt.title(f"Most expensive price taker for electricity in {year}")
        plt.legend(bbox_to_anchor=(1, 1))
        plt.savefig(f"{snakemake.output.price_taker}/{year}-price-taker.png", bbox_inches='tight')

    # plotting - market clearing price duration curve (price setter)
    data = results_s

    for year in planning_horizons:
        df = data[year].copy()
        # select only every 5th row
        df = df.iloc[::5, :]
        df.sort_values(by="marginal price @ bus", ascending=False, inplace=True)
        df.reset_index(inplace=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for carrier in df.carrier.unique():
            plt.plot(df[df.carrier == carrier]["marginal price @ bus"], 
                    marker='|', 
                    linestyle="", 
                    label=carrier, 
                    color=tech_colors[carrier])

        plt.ylim(-10, 250)
        plt.title(f"Price duration curve for electricity by price setter in {year}")
        plt.xlabel("Hours of the year in 3h steps (sorted)")
        plt.ylabel("Market clearing price in EUR/MWh)")
        plt.legend(bbox_to_anchor=(1, 1),fancybox=True, shadow=True, ncol=1)
        plt.savefig(f"{snakemake.output.pdc_price_setter}/{year}-pdc-price-setter.png", bbox_inches='tight')

    # plotting - market clearing price duration curve (price taker)
    data = results_d

    for year in years:
        df = data[year].copy()
        df = df[df.bidding_price.notna()]
        df.sort_values(by="marginal price @ bus", ascending=False, inplace=True)
        df.reset_index(inplace=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))

        for carrier in df.carrier.unique():
            plt.plot(df[df.carrier == carrier]["marginal price @ bus"], "*", label=carrier, color=tech_colors[carrier])

        plt.ylim(-10, 200)
        plt.title(f"Price duration curve for electricity by highest price taker in {year}")
        plt.xlabel("Hours of the year in 3h steps (sorted)")
        plt.ylabel("Market clearing price in EUR/MWh)")
        plt.legend(bbox_to_anchor=(1, 1),fancybox=True, shadow=True, ncol=1)
        plt.savefig(f"{snakemake.output.pdc_price_taker}/{year}-pdc-price-taker.png", bbox_inches='tight')

    # plotting - price duration curves
    # Fraction of time [%]
    bus = "DE0 0" 
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 5))

    for i, n in enumerate(networks.values()):
        lmps = pd.DataFrame(n.buses_t.marginal_price[bus])
        lmps.sort_values(by=bus, ascending=False, inplace=True)
        lmps["percentage"] = np.arange(len(lmps)) / len(lmps) * 100
        ax.plot(lmps["percentage"], lmps[bus], label=years[i], color=year_colors[i])

        ax.set_ylim([-50, 300])
        # # add corridor which contains 75 % of the generation around the median
        # ax.hlines(df["lmp"].loc[df["lmp"][df["gen_cumsum_norm"] > 0.125].index[0]], 0, 1, color=year_colors[i], ls="--", lw=1)
        # ax.hlines(df["lmp"].loc[df["lmp"][df["gen_cumsum_norm"] > 0.875].index[0]], 0, 1,  color=year_colors[i], ls="--", lw =1)

        ax.set_ylabel("Electricity Price [$€/MWh_{el}$")
        ax.set_xlabel("Fraction of time [%]")
        ax.set_title(f"Electricity price duration curves", fontsize=16)
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    plt.savefig(snakemake.output.elec_pdc, bbox_inches='tight')

