import os.path


NodeHeader = {
    "recording": "title,assetID,viewID",
    "isrc": "isrc",
    "composition": "share_asset_id,title",
    "artist": "name",
    "iswc": "iswc",
    "hfa_code": "code",
    "client": "name",
}

RelationshipHeader = {
    "owns": "start_id,end_id,custom_id,share,policy",
    "embedded": "start_id,end_id",
    "has_isrc": "start_id,end_id",
    "has_iswc": "start_id,end_id",
    "has_hfa_code": "start_id,end_id",
    "performed": "start_id,end_id",
    "wrote": "start_id,end_id",
}


def write_headers():
    for name, value in {**NodeHeader, **RelationshipHeader}.items():
        with open(os.path.join("graph-nodes-rels", f"{name}.csv"), "w") as fp:
            fp.write(f"{value}\n")
