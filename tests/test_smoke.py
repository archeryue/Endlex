def test_package_imports():
    import endlex

    assert hasattr(endlex, "Tracker")
    assert hasattr(endlex, "upload_checkpoint")


def test_server_app_constructs(tmp_path):
    from endlex.server.app import create_app

    app = create_app(tmp_path)
    assert app.title == "Endlex"
