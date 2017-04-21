"""
Key performance indicators for sewer sheds
"""
import pandas as pd
from helpers import subset

class SewerShedKPI(object):

    def __init__(self, net):

        """
        calculate shed-wide KPIs given a SewerNet object
        """

        #HYDROLOGIC
        nodes = net.nodes()
        self.shed_area_ac =  nodes.Shape_Area.sum() / 43560.0
        self.tc_min = nodes.tc.max()

        #wt avg C
        self.weighted_avg_c = [(nodes.runoff_coefficient * nodes.Shape_Area).sum()
                               / nodes.Shape_Area.sum()]

        #HYDRAULIC
        df = net.conduits()
        branches = df.loc[df.PIPE_TYPE == 'BRANCH']
        trunks = df.loc[df.PIPE_TYPE == 'TRUNK']
        other = df.loc[(df.PIPE_TYPE !='BRANCH') & (df.PIPE_TYPE !='TRUNK')]

        #create a dataframe with summary stats on capacity utilization
        hydr_dict = dict(
            all_sewers = dict(
                total_mi =     df.Shape_Leng.sum() / 5280,
                oversized_mi =  subset(df, 0, 0.50) / 5280,
                efficient_mi =  subset(df, 0.50, 1.0) / 5280,
                moderate_mi =  subset(df, 1.0, 2.0) / 5280,
                severe_mi =    subset(df, 2.0) / 5280,
                cap_frac_mean = df.capacity_fraction.mean(),
            ),
            trunks = dict(
                total_mi =     trunks.Shape_Leng.sum() / 5280,
                oversized_mi =  subset(trunks, 0, 0.50) / 5280,
                efficient_mi =  subset(trunks, 0.50, 1.0) / 5280,
                moderate_mi =  subset(trunks, 1.0, 2.0) / 5280,
                severe_mi =    subset(trunks, 2.0) / 5280,
                cap_frac_mean = trunks.capacity_fraction.mean(),
            ),
            branches = dict(
                total_mi =     branches.Shape_Leng.sum() / 5280,
                oversized_mi =  subset(branches, 0, .50) / 5280,
                efficient_mi =  subset(branches, .50, 1.0) / 5280,
                moderate_mi =  subset(branches, 1.0, 2.0) / 5280,
                severe_mi =    subset(branches, 2.0) / 5280,
                cap_frac_mean = branches.capacity_fraction.mean(),
            ),
            other = dict(
                total_mi =     other.Shape_Leng.sum() / 5280,
                oversized_mi =  subset(other, 0, .50) / 5280,
                efficient_mi =  subset(other, .50, 1.0) / 5280,
                moderate_mi =  subset(other, 1.0, 2.0) / 5280,
                severe_mi =    subset(other, 2.0) / 5280,
                cap_frac_mean = other.capacity_fraction.mean(),
            ),
        )
        index = ['total_mi','oversized_mi', 'efficient_mi', 'moderate_mi',
                 'severe_mi', 'cap_frac_mean']
        cols = ['all_sewers', 'branches', 'trunks', 'other']
        self.capacity = pd.DataFrame(hydr_dict, index = index, columns=cols)
