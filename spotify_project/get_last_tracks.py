import os
import spotify
import sqlite3
import pandas as pd


def main():
    # Get a dataframe with the last tracks
    print("Start to load the last tracks listened")
    last_tracks_df = spotify.get_recent_tracks(50)
    last_tracks_df.sort_values(by='played_at', ascending=False, inplace=True)
    print("Last tracks listened have been imported successfully !")

    conn = sqlite3.connect('spotify_project.db')

    # Retrieve existing played_at timestamps from the database for the 50 last tracks (because we can't fetch more than 50 tracks)
    query = "SELECT played_at FROM tracks_history ORDER BY played_at DESC LIMIT 50"
    # Try to fetch the last 50 'played_at' date tracks from the database
    existing_tracks = pd.read_sql(query, conn)

    # Keep only new tracks (those whose played_at is not already in the database)
    if existing_tracks.empty:
        new_tracks_df = last_tracks_df
    else:
        new_tracks_df = last_tracks_df[~last_tracks_df['played_at'].isin(existing_tracks['played_at'])]

    # Append only the truly new tracks to the DB
    if not new_tracks_df.empty:
        new_tracks_df.to_sql("tracks_history", conn, if_exists="append", index=False)
        print(f"{len(new_tracks_df)} new tracks have been added successfully to the Database !")
    else:
        print("No new tracks to add")

    conn.close()

if __name__ == "__main__":
    main()
    print("Done !")