--- asset full ---

select ASSET_ID, ASSET_TITLE, ISRC, ASSET_ARTIST, VIEW_ID, SHARE_ASSET_ID, ISWC, ASSET_SHARE_TITLE,
               ASSET_WRITERS, HFA_CODE, CLIENT
from db_orfium.yt_raw.ASSET_FULL_PUB
where report_date='2022-03-27'
order by SHARE_ASSET_ID

--- asset share ---

select SHARE_ASSET_ID, CUSTOM_ID, POLICY, OWNERSHIP_PROVIDED, CLIENT
from db_orfium.yt_raw.ASSET_SHARE
where report_date='2022-03-27' and ASSET_TYPE='ASSET_TYPE_COMPOSITION'
and contains(COPYRIGHT_TYPE, 'TYPE_SYNCH')