import pandas as pd
import numpy as np
import os
import sqlite3

if __name__ == "__main__":

    mapping_file = '../../../../Temp/MA/msd_song_mapping.csv'
    tag_sqlite_file = '../../../../Temp/MA/lastfm_tags.db'
    interaction_file = '../../../../Temp/MA/lastfm_tags.db'


    feature_file = '../../../../Temp/MA/song_features.npz'
    mapping =  pd.read_csv(mapping_file)
    tag_db = sqlite3.connect(tag_sqlite_file)
    msd_sample = pd.read_csv('../../../../Temp/MA/msd_data.csv')


    tags = pd.read_csv('../../../../Temp/MA/lastfm_unique_tags.tsv', sep='\t', header=None, names =['tag','frequency'])
    tag_song_df = pd.read_sql_query(" select tag, val, tids.tid from tid_tag, tids where tid_tag.tid = tids.ROWID", tag_db)
    tag_df = pd.read_sql_query("SELECT ROWID as tag_id,tag FROM tags", tag_db)
    relevant_tags = tag_df.merge(tags, on='tag').sort_values('frequency',ascending=False)
    relevant_tags.set_index('tag_id')
    relevant_tags = relevant_tags[:51]

    tag_song_df = tag_song_df.merge(relevant_tags, how='inner', left_on='tag', right_index=True)
    tag_song_df = tag_song_df[['tid', 'tag_id', 'val']]
    selected_tag_songs = tag_song_df.merge(mapping, on='tid', how='inner')

    song_features = pd.pivot_table(selected_tag_songs, values='val', index=['item_id'], columns=['tag_id'], fill_value=0).to_numpy()
    print("Features shape:", song_features.shape)

    nr_users = msd_sample.user_id.nunique()
    frequencies = msd_sample.pivot_table(values='user_id', index='item_id', aggfunc='count',fill_value=0) / nr_users

    np.savez(feature_file, features = song_features, known_frequency=frequencies)
