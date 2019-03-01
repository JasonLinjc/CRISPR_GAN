# -*- coding: utf-8 -*-
# @Time     :9/19/18 8:37 PM
# @Auther   :Jason Lin
# @File     :merge_penghui_dataset$.py
# @Software :PyCharm

import pandas as pd
import numpy as np
import random

nag_offs = pd.read_excel("penghui_dataset.xlsx", sheet_name='negative')
pos1_offs = pd.read_excel("penghui_dataset.xlsx", sheet_name='low throughput')
pos2_offs = pd.read_excel("penghui_dataset.xlsx", sheet_name='high throughput')
more_neg_idx = random.sample(range(30000, 50000), 5000)

merged_df0 = pd.DataFrame()
merged_df0['sgRNA'] = nag_offs['on-target site'].ix[more_neg_idx]
merged_df0['offtarget'] = nag_offs['no editing site'].ix[more_neg_idx]
merged_df0['label'] = np.zeros(len(merged_df0['offtarget']))

merged_df1 = pd.DataFrame()
merged_df1['sgRNA'] = nag_offs['on-target site'].ix[:20000]
merged_df1['offtarget'] = nag_offs['no editing site'].ix[:20000]
merged_df1['label'] = np.zeros(len(merged_df1['offtarget']))

merged_df2 = pd.DataFrame()
merged_df2['sgRNA'] = pos1_offs['on-target site']
merged_df2['offtarget'] = pos1_offs['off-target site']
merged_df2['label'] = np.ones(len(merged_df2['offtarget']))

merged_df3 = pd.DataFrame()
merged_df3['sgRNA'] = pos2_offs['on-target site']
merged_df3['offtarget'] = pos2_offs['off-target site']
merged_df3['label'] = np.ones(len(merged_df3['offtarget']))

"""
merged_df = pd.concat([merged_df1, merged_df2, merged_df2, merged_df3, merged_df3, merged_df2, merged_df3, merged_df0])
merged_df.to_csv("penghui_dataset_oversample1.csv", index=False)
print(merged_df)
"""

merged_df = pd.concat([merged_df2, merged_df3])
merged_df.to_csv("penghui_dataset_positive.csv", index=False)


