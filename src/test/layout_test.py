from src.frontend import show_clustering_gui_layouts


def test_clustering_layout():
    layout = show_clustering_gui_layouts("KMeans")
    assert layout is not None
    layout = show_clustering_gui_layouts("DBSCAN")
    assert layout is not None
    layout = show_clustering_gui_layouts("HDBSCAN")
    assert layout is not None
