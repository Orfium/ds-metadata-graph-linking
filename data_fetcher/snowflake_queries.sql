--- asset full ---

select ASSET_ID, ASSET_TITLE, ISRC, ASSET_ARTIST, VIEW_ID, SHARE_ASSET_ID, ISWC, ASSET_SHARE_TITLE,
               ASSET_WRITERS, HFA_CODE, CLIENT
from db_orfium.yt_raw.ASSET_FULL_PUB
where report_date='2022-03-27'
order by SHARE_ASSET_ID;

--- asset share ---

select SHARE_ASSET_ID, CUSTOM_ID, POLICY, OWNERSHIP_PROVIDED, CLIENT
from db_orfium.yt_raw.ASSET_SHARE
where report_date='2022-03-27' and ASSET_TYPE='ASSET_TYPE_COMPOSITION'
and contains(COPYRIGHT_TYPE, 'TYPE_SYNCH');

--- latest asset full ---
with cte_client_dates_last as (
    select client, max(report_date) as latest_date
    from db_orfium.yt_raw.asset_full_pub
    group by client
)
select ASSET_ID, ASSET_TITLE, ISRC, ASSET_ARTIST, VIEW_ID, SHARE_ASSET_ID, ISWC, ASSET_SHARE_TITLE,
               ASSET_WRITERS, HFA_CODE, afp.CLIENT, afp.report_date
from db_orfium.yt_raw.ASSET_FULL_PUB afp
inner join cte_client_dates_last cdl
on cdl.client=afp.client and afp.report_date=cdl.latest_date
order by SHARE_ASSET_ID;


--- latest rejects ---
with cte_client_dates_last as (
    select client, max(report_date) as latest_date
    from db_orfium.yt_raw.asset_full_pub
    group by client
)
select asf.ASSET_ID, asf.ASSET_TITLE, asf.ASSET_ARTIST,
    asf.ASSET_WRITERS, asm.COMPOSITION_TITLE, asm.COMPOSITION_WRITERS, asm.COMPOSITION_ID
from db_orfium.yt_raw.ASSET_FULL_PUB asf
inner join cte_client_dates_last cdl
on cdl.client=asf.client and asf.report_date=cdl.latest_date
inner join db_orfium.rm_public.asset_matching_resultassetmatch asm on asm.recording_id = asf.asset_id
where asm.STATUS = 'REJECTED'
order by SHARE_ASSET_ID;


--- latest_asset_share ---

with cte_client_dates_last as (
    select client, max(report_date) as latest_date
    from db_orfium.yt_raw.ASSET_SHARE
    group by client
)
select SHARE_ASSET_ID, CUSTOM_ID, POLICY, OWNERSHIP_PROVIDED, asf.CLIENT
from db_orfium.yt_raw.ASSET_SHARE asf
inner join cte_client_dates_last cdl
on cdl.client=asf.client and asf.report_date=cdl.latest_date
where asf.ASSET_TYPE='ASSET_TYPE_COMPOSITION' and contains(asf.COPYRIGHT_TYPE, 'TYPE_SYNCH');