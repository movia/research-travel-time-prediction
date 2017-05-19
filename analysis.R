library(readr)
library(data.table)
library(rgdal)
library(ggplot2)
library(ggthemes)
library(scales)
library(leaflet)
library(sp)
library(sf)

setwd('C:/Development/travel-time-prediction/')

read_results <- function(file) {
  results <- as.data.table(read_csv(file))
  results$Error = results$LinkTravelTime - results$LinkTravelTime_Predicted
  results$Hour <- as.numeric(format(results$DateTime, "%H"))
  results$DayType = factor(results$DayType)
  results$LinkName <- factor(results$LinkName, levels = unique(results$LinkName[order(results$LineDirectionLinkOrder)]))
  results$LinkOrder <- as.integer(results$LinkName)
  results
}

plot_result_errors <- function(ds = list(), labels = list()) {
  p <-  ggplot()
  
  for (i in seq_len(min(length(ds), length(labels)))) {
    loop_input = paste("stat_ecdf(data = ds[[i]], aes(x = Error, colour = '",labels[[i]],"'))", sep="")
    p <- p + eval(parse(text=loop_input)) 
  }

  p <- p +
    scale_y_continuous(labels=percent) + 
    facet_grid(LinkOrder ~ .) +
    theme_tufte() +
    theme(panel.grid = element_line(size = .25, linetype = "solid", color = "black")) +
    theme(legend.position = "bottom")
    
  p
}

results_lr_single <- read_results('./data/results_lr_single.csv')
results_lr_multiple <- read_results('./data/results_lr_multiple.csv')

plot_result_errors(list(results_lr_single, results_lr_multiple), list('LR single', 'LR multiple')) +
  xlim(-100, 150) +
  theme(axis.text.x = element_text(size=7)) +
  theme(axis.text.y = element_text(size=7)) +
  ggsave('plots/results_lr_errors.pdf', width = 210, height = 148, units = "mm")

results_svr_single <- read_results('./data/results_svr_single.csv')
results_svr_multiple <- read_results('./data/results_svr_multiple.csv')

plot_result_errors(list(results_svr_single, results_svr_multiple), list('SVR single', 'SVR multiple')) +
  xlim(-100, 150) +
  theme(axis.text.x = element_text(size=7)) +
  theme(axis.text.y = element_text(size=7)) +
  ggsave('plots/results_svr_errors.pdf', width = 210, height = 148, units = "mm")

# Look at some of the exstreame errors:
results_svr_single[abs(Error) > 120,]
results_svr_multiple[abs(Error) > 120,]

results_nn_single <- read_results('./data/results_nn_single.csv')
results_nn_multiple <- read_results('./data/results_nn_multiple.csv')

plot_result_errors(list(results_nn_single, results_nn_multiple), list('DNN single', 'DNN multiple')) +
  xlim(-100, 150) +
  theme(axis.text.x = element_text(size=7)) +
  theme(axis.text.y = element_text(size=7)) +
  ggsave('plots/results_dnn_errors.pdf', width = 210, height = 148, units = "mm")

results_nn_single[abs(Error) > 120,]
results_nn_multiple[abs(Error) > 120, list(JourneyLinkRef, DateTime, Observed, Predicted, Error)]







plot_result_errors(list(results_lr_single, results_svr_single, results_nn_single), list('LR single', 'SVR single', 'NN single')) +
  xlim(-100, 150) +
  theme(axis.text.x = element_text(size=7)) +
  theme(axis.text.y = element_text(size=7))

plot_result_errors(list(results_lr_multiple, results_svr_multiple, results_nn_multiple), list('LR multiple', 'SVR multiple', 'NN multiple')) +
  xlim(-100, 150) +
  theme(axis.text.x = element_text(size=7)) +
  theme(axis.text.y = element_text(size=7))

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
  scale_y_continuous(limits = quantile(d1_loi$LinkTravelTime, c(0.1, 0.9))) +
  labs(x = "Link Index", y = "Link travel time (s)") +
  theme_bw() +
  ggsave('plots/d1_loi_boxplot_nooutlier.pdf', width = 120, height = 80, units = "mm")

setkey(d1_loi, LinkTravelTime)
d1_loi_b10 <- d1_loi[, tail(.SD, 10), by=LinkName]
d1_loi_b5p <- d1_loi[, tail(.SD, 0.05 * .N), by=LinkName]

gplot(d1_loi_b5p)

d <- d1_loi_b5p[,.N, by=JourneyRef][,list(Count=sum(.N)),by=N][order(N),list(N,CumCount=cumsum(Count))]
ggplot(d, aes(N, CumCount)) + 
  geom_bar(stat = "identity") +
  #scale_y_continuous(labels=percent) +
  labs(x = "Number of links", y = "Cum. frequency") +
  theme_bw() +
  ggsave('plots/d1_loi_b5p_by_journeyref.pdf', width = 120, height = 80, units = "mm")


d1_loi_b5p$Hour <- as.numeric(format(d1_loi_b5p$DateTime, "%H")) + as.numeric(format(d1_loi_b5p$DateTime, "%M"))/60
ggplot(d1_loi_b5p, aes(Hour)) +
  geom_histogram(bins = 24 * 4) +
  theme_bw() +
  ggsave('plots/d1_loi_b5p_hour_histogram.pdf', width = 120, height = 80, units = "mm")
