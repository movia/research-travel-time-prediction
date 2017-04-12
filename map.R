library(RColorBrewer)
library(leaflet)
library(rgdal)

stop_points <- readOGR(dsn = "C:/Development/vehicletracker/mapmatcher/logs/expected_points_20160205L0004J0201.json", layer = "OGRGeoJSON", use_iconv = TRUE, encoding = "UTF-8")
expected_route <- readOGR(dsn = "C:/Development/vehicletracker/mapmatcher/logs/expected_route_4a_matched.json", layer = "OGRGeoJSON", use_iconv = TRUE, encoding = "UTF-8")

brewer.pal(9, 'Set1')

expected_route$color <- rep(brewer.pal(9, 'Set1'), length.out = length(expected_route))

m <- leaflet() %>%
  #addTiles() %>%
  addProviderTiles("CartoDB.Positron") %>%
  addPolylines(data = expected_route, weight = 3, opacity = 1, stroke = T, color = ~color, popup = ~as.character(sectionRef)) %>%
  addMarkers(data = stop_points[stop_points$isStopPoint == 1,], popup = ~as.character(stopPointName))
  
m


