library(RColorBrewer)
library(leaflet)
library(rgdal)

setwd('c:/development/travel-time-prediction')

route_links <- readOGR(dsn = "data/4A_RouteLinks.geojson", layer = "OGRGeoJSON", use_iconv = TRUE, encoding = "UTF-8")

stop_points <- readOGR(dsn = "C:/Development/vehicletracker/mapmatcher/logs/expected_points_20160205L0004J0201.json", layer = "OGRGeoJSON", use_iconv = TRUE, encoding = "UTF-8")
expected_route <- readOGR(dsn = "C:/Development/vehicletracker/mapmatcher/logs/expected_route_4a_matched.json", layer = "OGRGeoJSON", use_iconv = TRUE, encoding = "UTF-8")

brewer.pal(9, 'Set1')

expected_route$color <- rep(brewer.pal(9, 'Set1'), length.out = length(expected_route))


bus_icon <- makeIcon('graphics/bus.png', 'graphics/bus@2x.png', 16, 16)

m <- leaflet() %>%
  #addTiles() %>%
  addProviderTiles("CartoDB.Positron") %>%
  #addPolylines(data = expected_route, weight = 3, opacity = 1, stroke = T, color = ~color, popup = ~as.character(sectionRef)) %>%
  addPolylines(data = route_links, weight = 3, opacity = 1, stroke = T, popup = ~as.character(LinkName)) %>%
  addMarkers(data = stop_points[stop_points$isStopPoint == 1,], popup = ~as.character(stopPointName), icon = bus_icon)
  
m


