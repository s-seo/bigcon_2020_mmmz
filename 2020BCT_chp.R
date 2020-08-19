

library(readxl)
library(openxlsx)
library(stringr)
library(dplyr)
library(ggplot2)
library(reshape)

#------------------------------------------------------------------------------#
setwd('C:\\Users\\baoro\\Desktop\\공모전\\BCT_2020\\data\\01_제공데이터')

sale = read_excel('chp_sale.xlsx', 
                 1,trim_ws = TRUE) %>% 
  mutate(DATE = substr(방송일시, 1, 10),
         TIME = substr(방송일시, 12, 16)) %>% 
  .[,-1] %>% 
  as.data.frame()


rate = read.xlsx('chp_rate.xlsx',1)
rownames(rate) = rate[,1]
rate = rate[,-1]

wth = read.csv('C:\\Users\\baoro\\Desktop\\공모전\\BCT_2020\\data\\OBS_ASOS_TIM_20200807005933.csv')

names = list.files('C:\\Users\\baoro\\Desktop\\공모전\\BCT_2020\\data\\wth_fcst')
for(i in 1:length(names)){
  tmp = read.csv(paste0('C:\\Users\\baoro\\Desktop\\공모전\\BCT_2020\\data\\wth_fcst\\',
                        names[i]))
  assign(paste0('wth_fcst_',i), tmp)
}

sale$상품군 %>% table() %>% as.data.frame()

sale %>% group_by(상품명, 상품군) %>% summarise(n = tally()) %>% as.data.frame()

tmp = read.csv('C:\\Users\\baoro\\Desktop\\공모전\\BCT_2020\\data\\산업통상자원부_주요 유통업체 전년동월대비 매출증가율 추이_20200630.csv')
View(tmp)



load('C:\\Users\\baoro\\Desktop\\공모전\\BCT_2020\\data\\0806_chp.RData')
#------------------------------------------------------------------------------#


rate_group = rate[-nrow(rate),] %>% 
  mutate(seq = rep(1:((nrow(rate)-1)/20), each=20)) %>% 
  melt(., id.vars = 'seq') %>% 
  group_by(seq, variable) %>% 
  summarise(rate_mean = mean(value),
            rate_med = median(value),
            rate_max = max(value),
            rate_min = min(value),
            rate_sd = sd(value)) %>% 
  as.data.frame() %>% 
  arrange(variable)


mns <- function(m) {
  x <- m * 60
  return(x)
}

sale_rate = sale %>% 
  mutate(rate_mean = 0,
         rate_median = 0,
         rate_max = 0,
         rate_sd = 0)

for(i in 1:nrow(sale)){
  if(is.na(sale[i,1])|substr(sale[i,8],1,4) == '2020'){
    next
  }
  time_seq = strptime(sale[i,9], format = '%H:%M')  + mns(seq(1,round(sale[i,1]))-1)
  rate_seq = rate[rownames(rate) %in% substr(time_seq, 12,16), colnames(rate) == sale[i,8]]
  sale_rate$rate_mean[i] = mean(rate_seq,na.rm = T)
  sale_rate$rate_median[i] = median(rate_seq,na.rm = T)
  sale_rate$rate_max[i] = max(rate_seq,na.rm = T)
  sale_rate$rate_sd[i] = sd(rate_seq,na.rm = T)
}

View(sale_rate)

#-----------------------------------------------------------------------#
#--------- 0811 --------------------------------------------------------#
#-----------------------------------------------------------------------#

library(stringr)
library(httr)
library(XML)

api_url = "https://openapi.naver.com/v1/search/shop.xml"

query = URLencode(iconv("안드로이드", to="UTF-8"))
query = str_c("?query=", query)
client_id = '2lxwbBf9CLh83NXM9pkH'
client_secret = 'zPtWdrmCUt'
result = GET(str_c(api_url, query), 
             add_headers("X-Naver-Client-Id" = client_id, "X-Naver-Client-Secret" = client_secret))
xml_ = xmlParse(result)
xpathSApply(xml_, "/rss/channel/item/title", xmlValue)
xpathSApply(xml_, "/rss/channel/item/link", xmlValue)
xpathSApply(xml_, "/rss/channel/item/description", xmlValue)



result = GET("https://openapi.naver.com/v1/search/shop.xml?query=%EC%A3%BC%EC%8B%9D&display=10&start=1&sort=sim", 
             add_headers("X-Naver-Client-Id" = client_id, "X-Naver-Client-Secret" = client_secret))
xml_ = xmlParse(result)
xpathSApply(xml_, "/rss/channel/item/title", xmlValue)
xpathSApply(xml_, "/rss/channel/item/link", xmlValue)
xpathSApply(xml_, "/rss/channel/item/description", xmlValue)



library(jsonlite)
tmp = fromJSON('C:\\Users\\baoro\\Desktop\\공모전\\BCT_2020\\data\\naver_shopping_1.json')
tmp = gsub('\\"','',tmp) %>% 
  gsub('\\{','',.) %>% 
  gsub('\\}','',.) %>% 
  str_split(., ',') %>% 
  unlist() %>%
  .[-c(1:3)] %>% 
  matrix(., ncol=1096,byrow = TRUE)

tmp2 = cbind(rep(substr(tmp[1,1],16,19),547) %>% as.data.frame(),
      substr(tmp[1,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
      substr(tmp[1,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp2) = c('category','date','click_rate')
levels(tmp2[,2])[547] = '2019-01-01'

tmp3 = cbind(rep(substr(tmp[2,1],7,10),547) %>% as.data.frame(),
             substr(tmp[2,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[2,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp3) = c('category','date','click_rate')
levels(tmp3[,2])[547] = '2019-01-01'

tmp4 = cbind(rep(substr(tmp[3,1],7,12),547) %>% as.data.frame(),
             substr(tmp[3,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[3,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp4) = c('category','date','click_rate')
levels(tmp4[,2])[547] = '2019-01-01'

tmp_1 = rbind(tmp2,tmp3,tmp4)


tmp = fromJSON('C:\\Users\\baoro\\Desktop\\공모전\\BCT_2020\\data\\naver_shopping_2.json')
tmp = gsub('\\"','',tmp) %>% 
  gsub('\\{','',.) %>% 
  gsub('\\}','',.) %>% 
  str_split(., ',') %>% 
  unlist() %>%
  .[-c(1:3)] %>% 
  matrix(., ncol=1096,byrow = TRUE)

tmp2 = cbind(rep(substr(tmp[1,1],16,21),547) %>% as.data.frame(),
             substr(tmp[1,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[1,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp2) = c('category','date','click_rate')
levels(tmp2[,2])[547] = '2019-01-01'

tmp3 = cbind(rep(substr(tmp[2,1],7,13),547) %>% as.data.frame(),
             substr(tmp[2,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[2,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp3) = c('category','date','click_rate')
levels(tmp3[,2])[547] = '2019-01-01'

tmp4 = cbind(rep(substr(tmp[3,1],7,11),547) %>% as.data.frame(),
             substr(tmp[3,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[3,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp4) = c('category','date','click_rate')
levels(tmp4[,2])[547] = '2019-01-01'

tmp_2 = rbind(tmp2,tmp3,tmp4)



tmp = fromJSON('C:\\Users\\baoro\\Desktop\\공모전\\BCT_2020\\data\\naver_shopping_3.json')
tmp = gsub('\\"','',tmp) %>% 
  gsub('\\{','',.) %>% 
  gsub('\\}','',.) %>% 
  str_split(., ',') %>% 
  unlist() %>%
  .[-c(1:3)] %>% 
  matrix(., ncol=1096,byrow = TRUE)

tmp2 = cbind(rep(substr(tmp[1,1],16,17),547) %>% as.data.frame(),
             substr(tmp[1,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[1,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp2) = c('category','date','click_rate')
levels(tmp2[,2])[547] = '2019-01-01'

tmp3 = cbind(rep(substr(tmp[2,1],7,13),547) %>% as.data.frame(),
             substr(tmp[2,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[2,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp3) = c('category','date','click_rate')
levels(tmp3[,2])[547] = '2019-01-01'

tmp4 = cbind(rep(substr(tmp[3,1],7,11),547) %>% as.data.frame(),
             substr(tmp[3,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[3,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp4) = c('category','date','click_rate')
levels(tmp4[,2])[547] = '2019-01-01'

tmp_3 = rbind(tmp2,tmp3,tmp4)




tmp = fromJSON('C:\\Users\\baoro\\Desktop\\공모전\\BCT_2020\\data\\naver_shopping_4.json')
tmp = gsub('\\"','',tmp) %>% 
  gsub('\\{','',.) %>% 
  gsub('\\}','',.) %>% 
  str_split(., ',') %>% 
  unlist() %>%
  .[-c(1:3)] %>% 
  matrix(., ncol=1096,byrow = TRUE)

tmp2 = cbind(rep(substr(tmp[1,1],16,22),547) %>% as.data.frame(),
             substr(tmp[1,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[1,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp2) = c('category','date','click_rate')
levels(tmp2[,2])[547] = '2019-01-01'

tmp3 = cbind(rep(substr(tmp[2,1],7,9),547) %>% as.data.frame(),
             substr(tmp[2,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[2,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp3) = c('category','date','click_rate')
levels(tmp3[,2])[547] = '2019-01-01'

tmp_4 = rbind(tmp2,tmp3)
tmp = rbind(tmp_1,tmp_2,tmp_3,tmp_4)


tmp_m = merge(sale, tmp, by.x='DATE', by.y='date')
View(head(tmp_m,50))

tmp_m$상품군 %>% unique()
tmp_m$category %>% levels()

tmp_m[(tmp_m$상품군 == '의류') & (tmp_m$category == '패션의류'),] %>% 
  mutate(month = substr(DATE, 6,7) %>% as.factor()) %>% 
  group_by(click_rate,DATE,month) %>% 
  summarise(m = mean(취급액)) %>% 
  as.data.frame() %>% 
  ggplot(., aes(x=m, y=click_rate, color=month))+
  geom_point(size=2)+
  ggtitle('의류')


tmp_m[(tmp_m$상품군 == '농수축') & (tmp_m$category == '식품'),] %>% 
  mutate(month = substr(DATE, 6,7) %>% as.factor()) %>% 
  group_by(click_rate,DATE,month) %>% 
  summarise(m = mean(취급액)) %>% 
  as.data.frame() %>% 
  ggplot(., aes(x=m, y=click_rate, color=month))+
  geom_point(size=2)+
  ggtitle('농수축')


tmp_m[(tmp_m$상품군 == '생활용품') & (tmp_m$category == '여가/생활편의'),] %>% 
  mutate(month = substr(DATE, 6,7) %>% as.factor()) %>% 
  group_by(click_rate,DATE,month) %>% 
  summarise(m = mean(취급액)) %>% 
  as.data.frame() %>% 
  ggplot(., aes(x=m, y=click_rate, color=month))+
  geom_point(size=2)+
  ggtitle('생활용품')


tmp_m[(tmp_m$상품군 == '가전') & (tmp_m$category == '디지털/가전'),] %>% 
  mutate(month = substr(DATE, 6,7) %>% as.factor()) %>% 
  group_by(click_rate,DATE,month) %>% 
  summarise(m = mean(취급액)) %>% 
  as.data.frame() %>% 
  ggplot(., aes(x=m, y=click_rate, color=month))+
  geom_point(size=2)+
  ggtitle('가전')



View(rate)



