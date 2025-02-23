from flask import Flask, render_template, request
import modules.spotify as spotify

app = Flask(__name__)

tracks_df = None

@app.route('/')
def index():
    global tracks_df

    if tracks_df is None:
        playlist_id = '1z1tOO60TXJaLEfXb5Z1pw'

        try:
            print("Obtention de la playlist...")
            tracks_df = spotify.get_playlist_tracks(playlist_id)
            print("Playlist obtenue avec succès.")

        except Exception as e:
            print(f"Erreur lors de l'obtention de la playlist : {e}")
            tracks = []

    ### On convertit en une liste de dictionnaire pour bien être interprêté par Flask
    tracks = tracks_df.to_dict(orient='records')

    return render_template('index.html', tracks=tracks)

@app.route('/similar_tracks/<track_id>')
def similar_tracks(track_id):
    global tracks_df

    if tracks_df is None:
        return "Erreur : les données de la playlist n'ont pas été chargées."
    
    track = tracks_df[tracks_df['track_id'] == track_id]

    if track.empty:
        return "Aucun morceau trouvé avec cet ID."

    track_name = track['track_name'].iloc[0]

    artist_names = track['track_artists_name'].iloc[0].split(',')
    artist_name = artist_names[0].strip()

    print("track name : ", track_name, "de ", artist_name)

    ### Vectorize and scale tracks_df
    tracks_df_vect = spotify.vectorize_recent_tracks(tracks_df, n_components=10)

    weighted_features=['track_listeners', 'tracks_playcount', 'popularity', 'duration', 'release_date', 'album_id', 'track_tags', 'similar_artists', 'track_artists_id', 'album_artists_id']
    weights=[0, 0, 0.5, 0.2, 400, 1000, 10000, 1000, 1500, 1500]
    tracks_df_scaled = spotify.scale_and_weight(tracks_df_vect, weighted_features=weighted_features, weights=weights, n_components=10)

    ### Clustering
    clusters = spotify.clustering(tracks_df_scaled)
    tracks_df['cluster'] = clusters

    cluster_dfs = spotify.get_tracklist(tracks_df, limit=5)
    cluster_vect_dfs = spotify.vectorize_tracklist(cluster_dfs, weighted_features=weighted_features, weights=weights, n_components=1)
    cosine_matrices = spotify.compute_cosine_matrices(cluster_vect_dfs)

    similar_tracks = spotify.search_similar_tracks_by_name(tracks_df, cluster_dfs, cosine_matrices, track_name=track_name)
    similar_tracks_str = ", ".join(similar_tracks)
    print(f"tracks similaires à {track_name} : ", similar_tracks_str)

    similar_tracks_result = []

    for similar_track in similar_tracks:
        similar_tracks_result.append(spotify.search_track(similar_track))


    return render_template('similar_tracks.html', tracks=similar_tracks_result, original_track_name=track_name, artist_name=artist_name)

if __name__ == '__main__':
    app.run(debug=True)