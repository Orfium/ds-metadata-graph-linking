import os.path


NodeHeader = {
    "recording": "recording_title,assetID,viewID",
    "isrc": "isrc",
    "composition": "share_asset_id,composition_title",
    "artist": "name",
    "iswc": "iswc",
    "hfa_code": "code",
    "client": "client_name",
}

RelationshipHeader = {
    "owns": "client_name,share_asset_id,custom_id,share,policy",
    "embedded": "share_asset_id,assetID",
    "has_isrc": "assetID,isrc",
    "has_iswc": "share_asset_id,iswc",
    "has_hfa_code": "share_asset_id,hfa_code",
    "performed": "name,assetID",
    "wrote": "name,share_asset_id",
}


def write_headers(raw_data_path: str):
    for name, value in {**NodeHeader, **RelationshipHeader}.items():
        with open(os.path.join(raw_data_path, f"{name}.csv"), "w") as fp:
            fp.write(f"{value}\n")
