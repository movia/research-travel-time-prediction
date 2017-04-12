library(readr)
library(data.table)
library(rgdal)
library(ggplot2)
library(leaflet)
library(sp)
library(sf)

setwd('C:/Development/travel-time-prediction/')

#route_links <- readOGR(dsn = "data/4A_RouteLinks.csv", layer = "GeographyWkt", use_iconv = TRUE, encoding = "UTF-8")
#route_links1 <- st_read("data/4A_RouteLinks.csv", "4A_RouteLinks", crs = 4267)

#route_links <- read_csv("data/4A_RouteLinks.csv")
#route_links$Geography <- st_as_sfc(route_links$GeographyWkt)

data <- read_delim("./data/4A_201701.csv", 
         delim = ";",
         escape_double = FALSE,
         na = c("", "NA", "NULL"),
         col_types = cols(
           DateTime = col_datetime(format = "%Y-%m-%d %H:%M:%S"), 
           LinkTravelTime = col_integer()
         )
        )
setDT(data)

data <- data[LinkTravelTime > 0]
data$LinkName <- factor(data$LinkName, levels = unique(data$LinkName[order(data$LineDirectionCode, data$LineDirectionLinkOrder)]))

#levels(data$LinkName)$26

# Look only south bound
d1 <- data[LineDirectionCode == 1]
d1$LinkName <- droplevels(d1$LinkName)
d <- factor(unique(d1$LinkName))

d1_smry <- d1[, .N, by = list(LineDirectionLinkOrder, LinkRef, LinkName)][order(LineDirectionLinkOrder)]

# Select links of interest
d1_loi <- d1[(26 <= LineDirectionLinkOrder) & (LineDirectionLinkOrder <= 32)]
d1_loi$LinkName <- droplevels(d1_loi$LinkName)

ggplot(d1_loi, aes(factor(LineDirectionLinkOrder), LinkTravelTime)) + 
  geom_boxplot(outlier.shape = NA) +
  coord_cartesian(ylim = quantile(d1_loi$LinkTravelTime, c(0.1, 0.9))) +
  scale_y_continuous(limits = quantile(d1_loi$LinkTravelTime, c(0.1, 0.9)))

d1_loi[, head(.SD, 10), by=LinkName]