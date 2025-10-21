# Datasets

## 场景识别

### Places1

以场景为中心的数据集，其中包含 205 个场景类别和 250 万张带有类别标签的图像。

### Places205

一个图像数据集，其中包含来自 205 个场景类别的 2,448,873 张图像。
[下载链接](http://places.csail.mit.edu/downloadData.html)

### Places365(Places2)

Places2 是 Places 数据库的第 2 代，一个场景识别数据集，具有更多图像和场景类别。它由 1000 万张图像组成，包括 434 个场景类。

Places365 是 Places2 数据库的最新子集。
Places365-Standard 的列车集有来自 365 个场景类别的 ~180 万张图像，其中每个类别最多有 5000 张图像。
数据集有两个版本：
Places365-Standard， 训练集有来自 365 个场景类别的 ~180 万张图像，其中每个类别最多有 5000 张图像。
Places365-Challenge-2016，其中训练集还有额外的 620 万张图像以及 Places365-Standard 的所有图像（总共 ~800 万张图像），每个类别最多有 40,000 张图像。

This is the documentation of the Places365 data development kit. If you want the Places-CNN models instead of the training data, please refer to the [Places365-models](https://github.com/CSAILVision/places365).

**Downloads：**

注意：可以在[此处](http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar) 下载 Places365 标准 easyformat 拆分。

- First download the image list and annotations for [Places365-Standard](http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar) and the image list and annotations for [Place365-Challenge](http://data.csail.mit.edu/places/places365/filelist_places365-challenge.tar), and decompress the files in the data folder. This file only contains image list, without actual images.
- Download the corresponding compressed image files [at here](http://places2.csail.mit.edu/download.html). This file contains the actual images of Places database.

**Places365-Standard 数据集：**
Places365-Standard 的图像数据分为三种类型：来自 Places365-Standard 的训练数据 （TRAINING）、验证数据 （VALIDATION） 和测试数据 （TEST）。三个数据源没有重叠：TRAINING、VALIDATION 和 TEST。这三组数据都包含 365 类场景的图像。

Dataset | TRAIN | VALIDATION | TEST
--------|-------|------------|-----
Places365-Standard | 1,803,460 | 36,500 | 328,500

训练：between 3,068 and 5,000 per category
验证：100 images per category
测试：900 images per category

**Places365-Challenge 数据集：**
Places365-Challenge中使用的 365 个场景类别是 Places2 数据集的子集。
Places365-Challenge 和 Places365-Standard 之间的区别是 Places365-challenge 中还有 ~620 万多的额外图像。Places365-challenge 中前 5000 张（或更少）图像的类别属于 Places365-standard。
三个数据源没有重叠：TRAINING、VALIDATION 和 TEST。这三组数据都包含 365 类场景的图像。VALIDATION 和 TEST 与 Places365-Standard 相同。每个类别中的前 5000 张图像（或更少，因为它以该类别中的总图像数量为界）是来自 Places365-Standard 列车集的图像。

Dataset | TRAIN | VALIDATION | TEST
--------|-------|------------|-----
Places365-Challenge | 8,026,628 | 36,500 | 328,500

训练：between 3068 and 40,000 per category
验证：100 images per category
测试：900 images per category

**Places-Extra69 数据集：**
地点数据库中总共有 434 个场景类别。除了上面发布的 Places365 数据外，这里还发布了额外 69 个场景类别的数据。额外的 69 个类别的类别列表位于此处，其中每行包含场景类别名称，后跟其 ID（介于 0 和 68 之间的整数）。对于每个类别，我们保留 100 张图像作为测试图像。有 98,721 张图像用于训练，有 6,600 张图像用于测试。对于那些没有足够 100 张图像的类别，我们不会将它们包含在测试拆分中。
这个 Places-extra69 数据可能可用于单次学习或少样本学习，或一些迁移学习研究。

**标签列表：**

``` text
- airfield 0 机场
- airplane_cabin 1 飞机舱
- airport_terminal 2 机场航站楼
- alcove 3 壁龛
- alley 4 胡同
- amphitheater 5 圆形剧场
- amusement_arcade 6 游戏室
- amusement_park 7 游乐园
- apartment_building/outdoor 8 公寓楼/户外
- aquarium 9 水族馆
- aqueduct 10 渡槽
- arcade 11 拱廊
- arch 12 拱
- archaelogical_excavation 13 考古发掘
- archive 14 档案
- arena/hockey 15 竞技场/曲棍球
- arena/performance 16 竞技场/表演
- arena/rodeo 17 竞技场/竞技
- army_base 18 军队基地
- art_gallery 19 艺术画廊
- art_school 20 艺术学校
- art_studio 21 艺术工作室
- artists_loft 22 艺术家阁楼
- assembly_line 23 装配线
- athletic_field/outdoor 24 运动场/户外
- atrium/public 25 中庭/公共
- attic 26 阁楼
- auditorium 27 礼堂
- auto_factory 28 自动工厂
- auto_showroom 29 自动陈列室
- badlands 30 荒地
- bakery/shop 31 面包店/商店
- balcony/exterior 32 阳台/外部
- balcony/interior 33 阳台/室内
- ball_pit 34 球坑
- ballroom 35 舞厅
- bamboo_forest 36 竹林
- bank_vault 37 银行金库
- banquet_hall 38 宴会厅
- bar 39 酒吧
- barn 40 谷仓
- barndoor 41 谷仓门
- baseball_field 42 棒球场
- basement 43 地下室
- basketball_court/indoor 44 篮球场/室内
- bathroom 45 浴室
- bazaar/indoor 46 集市/室内
- b azaar/outdoor 47 集市/户外
- beach 48 海滩
- beach_house 49 海景房
- beauty_salon 50 美容院
- bedchamber 51 卧室
- bedroom 52 卧室
- beer_garden 53 啤酒花园
- beer_hall 54 啤酒馆
- berth 55 泊位
- biology_laboratory 56 生物实验室
- boardwalk 57 木板路
- boat_deck 58 船甲板
- boathouse 59 船库
- bookstore 60 书店
- booth/indoor 61 展位/室内
- botanical_garden 62 植物园
- bow_window/indoor 63 弓窗/室内
- boling_alley 64 保龄球馆
- boxing_ring 65 拳赛场地
- bridge 66 桥
- building_facade 67 建筑立面
- bullring 68 斗牛场
- burial_chamber 69 墓室
- bus_interior 70 巴士内部
- bus_station/indoor 71 公交车站/室内
- butchers_shop 72 肉店
- butte 73 比尤特
- cabin/outdoor 74 机舱/户外
- cafeteria 75 自助餐厅
- campsite 76 营地
- campus 77 校园
- canal/natural 78 运河/自然
- canal/urban 79 运河/城市
- candy_store 80 糖果店
- canyon 81 峡谷
- car_interior 82 汽车内饰
- carrousel 83 旋转木马
- castle 84 城堡
- catacomb 85 地下墓穴
- cemetery 86 公墓
- chalet 87 小木屋
- chemistry_lab 88 化学实验室
- childs_room 89 儿童房
- church/indoor 90 教堂/室内
- church/outdoor 91 教堂/户外
- classroom 92 课堂
- clean_room 93 整理房间
- cliff 94 悬崖
- closet 95 壁橱
- clothing_store 96 服装店
- coast 97 海岸
- cockpit 98 座舱
- coffee_shop 99 咖啡馆
- computer_room 100 计算机房
- conference_center 101 会议中心
- conference_room 102 会议室
- construction_sizte 103 施工现场
- corn_fild 104 玉米田
- corral 105 畜栏
- corridor 106 走廊
- cottage 107 小屋
- courthouse 108 法院大楼
- courtyard 109 庭院
- creek 110 溪
- crevasse 111 裂缝
- crosswalk 112 人行横道
- dam 113 坝
- delicatessen 114 熟食
- department_store 115 百货商店
- dsert/sand 116 沙漠/沙子
- desert/vegetation 117 沙漠/植被
- desert_road 118 沙漠路
- diner/outdoor 119 晚餐/户外
- dining_hall 120 餐厅
- dinging_room 121 饭厅
- discotheque 122 迪斯科舞厅
- doorway/outdoor 123 门口/户外
- dorm_room 124 宿舍
- downtown 125 市中心 
- dressing_room 126 更衣室
- driveway 127 车道
- drugstore 128 药店
- elevator/door 129 电梯/门
- elevator_lobby 130 电梯大堂
- elevator_shaft 131 电梯井
- embassy 132 大使馆
- engine_room 133 机房
- entrance_hall 134 入口大厅
- escalator/indoor 135 自动扶梯/室内
- excavation 136 挖掘
- fabric_store 137 面料商店
- farm 138 农场
- fastfood_restaurant 139 快餐餐厅
- field/cultivated 140 田地/栽培
- field/wild 141 野外/野生
- field_road 142 田间小路
- fire_escape 143 火灾逃生
- fire_station 144 消防站
- fishpond 145 鱼池
- flea_market/indoor 146 跳蚤市场/室内
- florist_shop/indoor 147 花店/室内
- food_court 148 美食广场
- footvall_field 149 足球场
- forest/broadleaf 150 森林/阔叶
- forest_path 151 森林路径
- forest_road 152 森林路
- formal_garden 153 正式花园
- fountain 154 喷泉
- galley 155 厨房
- garage/indoor 156 车库/室内
- garage/outdoor 157 车库/室外
- gas_station 158 加油站
- gazebo/exterioir 159 凉亭/外部
- general_store/indoor 160 杂货店/室内
- general_store/outdoor 161 杂货店/户外
- gift_shop 162 礼品店
- glacier 163 冰川
- golf_course 164 高尔夫球场
- greenhouse/indoor 165 温室/室内
- greenhouse/outdoor 166 温室/室外
- grotto 167 石窟
- gymnasium/indoor 168 健身房/室内
- hangar/indoor 169 机库/室内
- hangar/outdoor 170 机库/室外
- harbor 171 海港
- harware_store 172 五金店
- hayfield 173 海菲尔德
- heliport 174 直升机场
- highway 175 高速公路
- home_office 176 家庭办公室
- home_theater 177 家庭影院
- hospital 178 医院
- hospital_room 179 病房
- hot_sprint 180 热冲刺
- hotel/outdoor 181 酒店/户外
- hotel_room 182 酒店房间
- house 183 屋
- hunting_lodge/outdoor 184 狩猎小屋/户外
- ice_cream_parlor 185 冰淇淋店
- ice_floe 186 浮冰
-  ice_shelf 187 冰架
- ice_skating_rink/indoor 188 溜冰场/室内
- ice_skating_rink/outdoor 189 溜冰场/户外
- iceberg 190 冰山
- igloo 191 冰屋
- industrial_area 192 工业区
- inn/outdoor 193 旅馆/室外
- islet 194 胰岛
- jacuzzi/indoor 195 按摩浴缸/室内
- jail_cell 196 牢房
- japanese_garden 197 日本花园
- jewelry_shop 198 珠宝店
- junkyard 199 垃圾场
- kasbah 200 古堡
- kennel/outdoor 201 狗舍/户外
- kindergarden_classroom 202 幼儿园教室
- kitchen 203 厨房
- lagoon 204 泻湖
- lake/natural 205 湖水/天​​然
- landfill 206 垃圾填埋场
- landing_deck 207 登陆甲板
- laundromat 208 自助洗衣店
- lawn 209 草坪
- lecture_room 210 演讲厅
- legislative_chamber 211 立法会议厅
- library/indoor 212 图书馆/室内
- library/outdoor 213 图书馆/户外
- lighthouse 214 灯塔
- living_room 215 客厅
- loading_dock 216 装卸码头
- lobby 217 大堂
- lock_chamber 218 锁室
- locker_room 219 更衣室
- mansion 220 大厦
- manufactured_home 221 预制房屋
- market/indoor 222 市场/室内
- market/outdoor 223 市场/户外
- marsh 224 沼泽
- martial_arts_gym 225 武术体育馆
- mausoleum 226 陵墓
- medina 227 麦地那
- mezzanine 228 夹层
- moat/water 229 护城河/水
- mosque/outdoor 230 清真寺/户外
- motel 231 汽车旅馆
- mountain 232 山
- moutain_path 233 山路
- moutain_snowy 234 雪山
- movie_theater/indoor 235 电影剧院/室内
- museum/indoor 236 博物馆/室内
- museum/outdoor 237 博物馆/户外
- music_studio 238 音乐工作室
- natural_history_museum 239 自然历史博物馆
- nursery 240 托儿所
- nursing_home 241 护理之家
- oast_house 242 烤面包房
- ocean 243 海洋
- office 244 办公室
- office_building 245 办公楼
- office_cubicles 246 办公室隔间
- oilrig 247 石油钻井平台
- operating_room 248 手术室
- orchard 249 果园
- orchestra_pit 250 乐池
- pagoda 251 宝塔
- palace 252 宫殿
- pantry 253 储藏室
- park 254 公园
- parking_garage/indoor 255 停车库/室内
- parking_garage/outdoor 256 停车库/室外
- parking_lot 257 停车场
- pasture 258 牧场
- patio 259 露台
- pavilion 260 展馆
- pet_shop 261 宠物店
- pharmacy 262 药房
- phone_booth 263 电话亭
- physics_laboratory 264 物理实验室
- picnic_area 265 野餐区
- pier 266 码头
- pizzeria 267 比萨店
- playground 268 游乐场
- playroom 269 游戏室
- plaza 270 广场
- pond 271 池塘
- porch 272 门廊
- promenade 273 长廊
- pub/indoor 274 酒吧/室内
- raceourse 275 赛马场
- raceway 276 跑道
- raft 277 木筏
- railroad_track 278 铁轨
- rainforest 279 雨林
- receptiion 280 接待处
- recreatioin_room 281 娱乐室
- repair_shop 282 维修店
- residential_neighborhood 283 居民区
- restaurant 284 餐厅
- restaurant_kitchen 285 餐厅厨房
- restaurant_patio 286 餐厅露台
- rice_paddy 287 稻田
- river 288 河
- rock_arch 289 岩石拱门
- roof_garden 290 屋顶花园
- rope_bridge 291 索桥
- ruin 292 废墟
- runnway 293 跑道
- sandbox 294 沙箱
- sauna 295 桑拿
- schoolhouse 296 校舍
- science_museum 297 科学博物馆
- server_room 298 服务器机房
- shed 299 棚
- shopfront 301 店面
- shopping_mall/indoor 302 商场/室内
- shower 303 淋浴
- ski_resort 304 滑雪胜地
- ski_slope 305 滑雪坡
- sky 306 天空
- skyscraper 307 摩天大楼
- slum 308 贫民窟
- snowfield 309 雪原
- soccer_field 310 足球场
- stable 311 稳定的
- stadium/baseball 312 体育场/棒球
- stadium/football 313 体育场/足球
- stadium/soccer 314 体育场/足球
- stage/indoor 315 舞台/室内
- stage/outdoor 316 舞台/户外
- staircase 317 楼梯
- storage_room 318 储藏室
- street 319 街道
- subway_station/platform 320 地铁站/站台
- supermarket 321 超市
- sushi_bar 322 寿司吧
- swamp 323 沼泽
- swimming_hole 324 游泳洞
- swimming_pool/indoor 325 游泳池/室内
- swimming_pool/outdoor 326 游泳池/室外
- synagogue/outdoor 327 犹太教堂/户外
- television_room 331 电视室
- ticket_booth 332 售票亭
- topiary_garden 333 修剪花园
- tower 334 塔
- toyshop 335 玩具店
- train_interior 336 火车内部
- train_station/platform 337 火车站/平台
- tree_farm 338 林场
- tree_house 339 树屋
- trench 340 战壕
- tundra 341 苔原
- underwater/ocean_deep 342 水下/海洋深处
- utility_room 343 杂物间
- valley 344 山谷
- vegetable_garden 345 蔬菜园
- veterinarians_office 346 兽医办公室
- viaduct 347 高架桥
- village 348 村
- vineyard 349 葡萄园
- volcano 350 火山
- volleyball_court/outdoor 351 排球场/室外
- waiting_room 352 候车室
- water_park 353 水上乐园
- water_tower 354 水塔
- waterfall 355 瀑布
- watering_hole 356 水坑
- wave 357 海浪
- wet_bar 358 湿棒
- wheat_field 359 麦田
- wind_farm 360 风电场
- windmill 361 风车
- yard 362 码
- youth_hostel 363 青年旅舍
- zen_garden 364 禅园
```

``` text
- toll_plaza 0 收费站
- baggage_claim 1 行李领取
- dentists_office 2 牙医办公室
- lido_deck/outdoor 3 丽都甲板/户外
- hot_tub/outdoor 4 热水浴缸/户外
- dining_car 5 餐车
- videostore 6 音像店
- cheese_factory 7 奶酪工厂
- courtroom 8 法庭
- elevator/interior 9 电梯/室内
- great_hall 10 大厅
- teashop 11 茶馆
- labyrinth/outdoor 12 迷宫/户外
- ranch_house 13 牧场屋
- promenade_deck 14 长廊甲板
- warehouse/indoor 15 仓库/室内
- volleyball_court/indoor 16 排球场/室内
- music_store 17 音乐商店
- limousine_interior 18 豪华轿车内饰
- wine_cellar/barrel_storage 19 酒窖/桶存储
- tennis_court/outdoor 20 网球场/室外
- firing_range/indoor 21 射程/室内
- tennis_court/indoor 22 网球场/室内
- casino/indoor 23 赌场/室内
- subway_interior 24 地铁内部
- wine_cellar/bottle_storage 25 酒窖/酒瓶储藏室
- badminton_court/indoor 26 羽毛球场/室内
- optician 27 配镜师
- basketball_court/outdoor 28 篮球场/室外
- squash_court 29 壁球场
- hat_shop 30 帽子店
- athletic_field/indoor 31 运动场/室内
- cybercafe 32 网吧
- loft 33 阁楼
- electrical_substation 34 变电站
- thriftshop 35 旧货店
- factory/indoor 36 工厂/室内
- shower_room 37 淋浴房
- bow_window/outdoor 38 弓窗/户外
- nuclear_power_plant/outdoor 39 核电站/室外
- anechoic_chamber 40 消声室
- batters_box 41 击球手框
- chicken_coop/outdoor 42 鸡舍/户外
- hot_tub/indoor 43 热水浴缸/室内
- editing_room 44 编辑室
- observatory/outdoor 45 天文台/户外
- dinette/vehicle 46 餐桌椅/车
- rest_area 47 休息区
- portrait_studio 48 肖像工作室
- covered_bridge/exterior 49 廊桥/外部
- funeral_home 50 殡仪馆
- kennel/indoor 51 狗舍/室内
- power_plant/outdoor 52 发电厂/户外
- walk_in_freezer 53 走进冷冻室
- oil_refinery/outdoor 54 炼油厂/户外
- forest/needleleaf 55 森林/针叶
- florist_shop/outdoor 56 花店/户外
- liquor_store/outdoor 57 酒类商店/户外
- jail/indoor 58 监狱/室内
- poolroom/home 59 台球室/家庭
- driving_range/outdoor 60 练习场/户外
- brewery/indoor 61 啤酒厂/室内
- outhouse/outdoor 62 室外/室外
- podium/indoor 63 讲台/室内
- theater/indoor_seats 64 剧院/室内座位
- fitting_room/interior 65 试衣间/室内
- aquatic_theater 66 水上剧场
- podium/outdoor 67 领奖台/户外
- synagogue/indoor 68 犹太教堂/室内
```

### MIT Indoor 67 Dataset

[MITndoor67数据集](https://github.com/vpulab/Semantic-Aware-Scene-Recognition/tree/master/Data/Datasets/MITIndoor67)
该数据库包含 67 个室内类别，共 15620 张图像。图像数量因类别而异，但每个类别至少有 100 张图像。所有图像均为 jpg 格式。
训练集和验证集之间的划分：每个场景类 80% 训练图像和 20% 用于验证的图像。
室内场景识别是高级视觉中具有挑战性的开放问题。大多数适用于室外场景的场景识别模型在室内域中表现不佳。主要困难在于，虽然一些室内场景（例如走廊）可以通过全局空间属性很好地表征，但其他场景（例如书店）则可以通过它们包含的对象更好地表征。更一般地说，为了解决室内场景识别问题，我们需要一个能够利用局部和全局判别信息的模型。

### SUN 397 Dataset

该数据库包含来自以下论文中使用的场景识别 SUN 数据集的 397 个类别子集。 图像数量因类别而异，但每个类别至少有 100 张图像，总共 108,754 张图像。 所有图像均为 jpg 格式。

---

## 人类动作识别

[UCF系列](https://www.crcv.ucf.edu/data/UCF101.php)。UCF101是由美国中佛罗里达大学计算机视觉研究中心创建的大型人类动作数据集，包含101种动作类别，共13320个视频片段，总时长约27小时。数据集内容丰富，涵盖了体育、乐器演奏、人类互动等多种场景，视频来源于YouTube，具有真实环境下的摄像头移动和复杂背景。创建过程中，研究人员从YouTube下载视频，手动筛选去除无关内容，确保数据集的质量。UCF101主要用于动作识别研究，旨在解决复杂环境下的动作分类问题，是目前最具挑战性的动作识别数据集之一。

## AI剪辑中的场景识别设想

### 1. 按拍摄环境和剪辑用途分类

- **人物对话场景**
  - 室内对话：客厅、办公室、咖啡厅、餐厅等
  - 室外对话：公园、街道、阳台等
  
- **独白/旁白场景**
  - 室内独处：卧室、书房、浴室等私密空间
  - 室外沉思：海边、山顶、花园等开阔环境

- **动作/情节场景**
  - 运动场景：体育馆、运动场、健身房等
  - 工作场景：厨房、工作室、办公区等
  - 交通工具：汽车内、火车、飞机等

- **过渡/空镜头场景**
  - 自然风光：山水、森林、海洋、天空等
  - 城市景观：街道、建筑、夜景等
  - 细节特写：花草、物品等特写镜头

- **社交聚会场景**
  - 庆祝场所：宴会厅、酒吧、派对场地等
  - 公共场所：商场、广场、公园等人群聚集地

- **特殊情境场景**
  - 医疗场景：医院、诊所等
  - 教育场景：教室、图书馆等
  - 紧急情况：火灾、事故现场等(如果适用)

### 2. 按镜头语言和视觉风格分类

- **静态场景**：适合长镜头、固定镜头的安静场景
- **动态场景**：适合快切、运动镜头的活跃场景
- **情感场景**：能传达特定情绪的场景(温馨、紧张、浪漫等)
- **展示场景**：用于产品展示、环境介绍的场景

## 实施策略

### 分类层次设计

``` text
主分类 (6-8类) → 子分类 (15-20类) → 原始类别 (365类)
```

## 主分类体系

### 1. 居住与生活空间 (Living & Lifestyle)

- **私人居住空间**
  - bedroom 卧室
  - bedchamber 卧室
  - living_room 客厅
  - childs_room 儿童房
  - attic 阁楼
  - basement 地下室
  - closet 壁橱
  - dressing_room 更衣室
  - bathroom 浴室
  - shower 淋浴
  - jacuzzi/indoor 按摩浴缸/室内

- **共享居住设施**
  - dorm_room 宿舍
  - apartment_building/outdoor 公寓楼/户外
  - house 屋
  - cottage 小屋
  - mansion 大厦
  - manufactured_home 预制房屋
  - ranch_house 牧场屋
  - chalet 小木屋

- **生活服务场所**
  - kitchen 厨房
  - dining_room 饭厅
  - dining_hall 餐厅
  - cafeteria 自助餐厅
  - diner/outdoor 晚餐/户外
  - restaurant 餐厅
  - restaurant_kitchen 餐厅厨房
  - restaurant_patio 餐厅露台
  - coffee_shop 咖啡馆
  - bakery/shop 面包店/商店
  - ice_cream_parlor 冰淇淋店
  - bar 酒吧
  - pub/indoor 酒吧/室内
  - beer_garden 啤酒花园
  - beer_hall 啤酒馆
  - banquet_hall 宴会厅

### 2. 工作与学习场所 (Work & Education)

- **办公场所**
  - office 办公室
  - office_building 办公楼
  - office_cubicles 办公室隔间
  - home_office 家庭办公室
  - conference_center 会议中心
  - conference_room 会议室
  - meeting_room 会议室

- **教育机构**
  - classroom 课堂
  - lecture_room 演讲厅
  - schoolhouse 校舍
  - campus 校园
  - library/indoor 图书馆/室内
  - library/outdoor 图书馆/户外
  - kindergarden_classroom 幼儿园教室

- **专业工作场所**
  - laboratory 实验室 (biology_laboratory, chemistry_lab, physics_laboratory)
  - studio 工作室 (art_studio, music_studio, portrait_studio)
  - workshop 工作坊 (engine_room, repair_shop, assembly_line)
  - medical_facility 医疗场所 (hospital, hospital_room, operating_room)
  - computer_room 计算机房
  - server_room 服务器机房

### 3. 商业与服务设施 (Commercial & Services)

- **零售商业**
  - store 商店 (general_store/indoor, general_store/outdoor)
  - specialty_shops 专业店 (butchers_shop, candy_store, clothing_store, fabric_store, florist_shop/indoor, florist_shop/outdoor, jewelry_shop, pet_shop, pharmacy, shoe_shop, toyshop, electronics_store, bookstore, gift_shop)
  - department_store 百货商店
  - shopping_mall/indoor 商场/室内
  - market 市场 (market/indoor, market/outdoor, flea_market/indoor, bazaar/indoor, bazaar/outdoor)
  - auto_showroom 自动陈列室

- **餐饮服务**
  - fastfood_restaurant 快餐餐厅
  - food_court 美食广场
  - delicatessen 熟食

- **其他服务**
  - beauty_salon 美容院
  - laundromat 自助洗衣店
  - gas_station 加油站
  - parking_garage 停车库 (indoor/outdoor)
  - parking_lot 停车场

### 4. 娱乐与休闲场所 (Entertainment & Recreation)

- **文化娱乐**
  - theater 剧院 (movie_theater/indoor, amphitheater)
  - art_gallery 艺术画廊
  - museum 博物馆 (art_museum, natural_history_museum, science_museum)
  - auditorium 礼堂
  - concert_hall 音乐厅

- **体育运动**
  - gymnasium/indoor 健身房/室内
  - sports_courts 运动场 (basketball_court/indoor, tennis_court/indoor, badminton_court/indoor, squash_court, volleyball_court/indoor)
  - sports_fields 运动场 (athletic_field/outdoor, baseball_field, football_field, soccer_field)
  - swimming_pool 游泳池 (indoor/outdoor)
  - golf_course 高尔夫球场
  - bowling_alley 保龄球馆
  - boxing_ring 拳赛场地

- **游乐休闲**
  - amusement_park 游乐园
  - amusement_arcade 游戏室
  - playground 游乐场
  - arcade 拱廊
  - botanical_garden 植物园
  - garden 花园 (formal_garden, japanese_garden, topiary_garden, zen_garden)
  - zoo 动物园 (虽然未在列表中，但可归入此类)

### 5. 交通与运输设施 (Transportation)

- **航空运输**
  - airfield 机场
  - airport_terminal 机场航站楼
  - airplane_cabin 飞机舱

- **陆路交通**
  - road 道路 (highway, street, alley, driveway, field_road, desert_road)
  - railway 铁路 (railroad_track, train_station/platform, train_interior)
  - bus_station 公交车站/室内
  - bus_interior 巴士内部
  - subway 地铁 (subway_station/platform, subway_interior)
  - parking 设施 (parking_garage, parking_lot)
  - toll_plaza 收费站

- **水上交通**
  - harbor 海港
  - boat_deck 船甲板
  - boathouse 船库

### 6. 宗教与政府建筑 (Religious & Government)

- **宗教场所**
  - church 教堂 (church/indoor, church/outdoor)
  - mosque 清真寺 (mosque/outdoor)
  - synagogue 犹太教堂 (synagogue/indoor, synagogue/outdoor)
  - temple 寺庙 (虽然未在列表中，但可归入此类)

- **政府机构**
  - courthouse 法院大楼
  - legislative_chamber 立法会议厅
  - city_building 市政建筑 (capitol_building虽然未在列表中)
  - embassy 大使馆

### 7. 自然与户外环境 (Nature & Outdoor)

- **自然景观**
  - mountain 山 (mountain, mountain_path, mountain_snowy)
  - forest 森林 (forest/broadleaf, forest_path, forest_road)
  - water_bodies 水体 (ocean, sea, lake/natural, river, creek, pond, waterfall, fountain, swimming_hole)
  - desert 沙漠 (desert/sand, desert/vegetation)
  - canyon 峡谷
  - glacier 冰川
  - volcano 火山
  - field 田野 (field/cultivated, field/wild, corn_field, hayfield, rice_paddy, wheat_field, pasture)
  - beach 海滩
  - sky 天空
  - garden 花园 (botanical_garden, formal_garden, japanese_garden, topiary_garden, zen_garden, vegetable_garden)

- **户外休闲**
  - park 公园
  - campsite 营地
  - picnic_area 野餐区
  - patio 露台
  - courtyard 庭院
  - plaza 广场
  - downtown 市中心
  - residential_neighborhood 居民区

### 8. 特殊用途场所 (Special Purpose)

- **工业设施**
  - factory 工厂 (auto_factory, cheese_factory)
  - industrial_area 工业区
  - construction_site 施工现场
  - power_plant 发电厂 (nuclear_power_plant/outdoor)
  - oil_refinery 炼油厂/outdoor
  - warehouse/indoor 仓库/室内

- **军事与安全**
  - army_base 军队基地
  - jail 监狱 (jail_cell, jail/indoor)
  - firing_range/indoor 射程/室内

- **其他特殊场所**
  - barn 谷仓
  - castle 城堡
  - catacomb 地下墓穴
  - cemetery 公墓
  - dam 坝
  - landfill 垃圾填埋场
  - junkyard 垃圾场
  - lighthouse 灯塔
  - observatory/outdoor 天文台/户外
  - viaduct 高架桥
  - windmill 风车
  - wind_farm 风电场

这种分类方式可以帮助AI剪辑系统根据场景类型选择合适的剪辑策略，例如：

- 居住空间场景适合温馨、私密的剪辑风格
- 工作场所场景适合专业、高效的剪辑节奏
- 娱乐休闲场景适合活泼、有趣的剪辑方式
- 自然景观场景适合舒缓、优美的剪辑节奏
