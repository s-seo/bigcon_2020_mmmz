# bigcon_2020_mmmz

### Variables (2020.08.28 / 37)

name | descript | tyoe 
---- | ---- | ---- 
index | index | index 
방송일시 | raw | object 
노출.분. | raw | object
마더코드 | raw | object
상품코드 | raw | object
상품명 | raw | object
상품군 | raw | object
판매단가 | raw | float
취급액 | raw | float
selling_price | raw(eng) | float
sales | raw(eng) | float
exposed.t | raw(eng) | float
volume | sales/selling_price | float
month | 월 | object(12)
day | 일 | object(31)
weekday | 요일 | object(7)
holiday | red+weekend | object(2)
red | 공휴일 | object(2)
weekend | 주말 | object(2)
hours | 시간 | object(24)
hours_inweek | 주 단위 시간 | object(168)
**primetime** | 1 = 오전 프라임, 2 = 오후 프라임 | object(3)
**japp** | 30분에서 반올림 | object()
**min_start** | 방송 시작시간(분) | object(8)
**min_range** | 방송 길이(분) | object()
**parttime** | 한 타임 내 방송순서 | object()
**show_id** | 해당 일 방송회차 | object()
**sales_power** | 시간 대비 판매량(?) | float
**men** | 성별 가지는 상품(null) | object(2)
**hottest** | 탑10 포함여부 | object(2)
**pay** | 지불방식 | object(2)
**luxury** | 명품 단가 50만 이상 | object(2)
**brandpower** | 업데이트 예정 | -
**season_item** | 업데이트 예정 | -
**naver_crawl** | 업데이트 예정 | -
**weather** | 업데이트 예정 | -
original_c | naver | object
small_c | naver | object
middle_c | naver(null) | object
big_c | naver(null) | object
brand | 브랜드(null) | object

