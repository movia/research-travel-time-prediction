with
[LinkTravelTime] as
(
    select
        [LinkRef] = concat(sec.StopPointSectionFromOwner, ':', sec.StopPointSectionFromNumber, '->', sec.StopPointSectionToOwner, ':',sec.StopPointSectionToNumber),        
        [LinkName] = sec.StopPointSectionDisplayName,
        sec.StopPointSectionGeographyId
    from
        [data].[RT_Journey] j
        join [data].[RT_JourneyPoint] p on p.[JourneyRef] = j.[JourneyRef]
        join [dim].[JourneyPatternSection] sec on sec.JourneyPatternId = j.[JourneyPatternId] and sec.SequenceNumber = p.SequenceNumber and sec.IsCurrent = 1
    where
        j.[LineDesignation] = '4A'
        and j.[OperatingDayDate] between '2017-01-01' and '2017-01-31'
        and p.[IsStopPoint] = 1
    group by 
        concat(sec.StopPointSectionFromOwner, ':', sec.StopPointSectionFromNumber, '->', sec.StopPointSectionToOwner, ':',sec.StopPointSectionToNumber),
        sec.StopPointSectionDisplayName,
        StopPointSectionGeographyId
)
select   
    [LinkRef],
    [LinkName],
    sec.Geography.ToString()
from
    [LinkTravelTime]
    join data.GIS_StopPointSection sec on sec.Id = StopPointSectionGeographyId
