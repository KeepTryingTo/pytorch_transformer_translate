---
annotations_creators:
- found
language_creators:
- found
language:
- ca
- de
- el
- en
- eo
- es
- fi
- fr
- hu
- it
- nl
- 'no'
- pl
- pt
- ru
- sv
license:
- other
multilinguality:
- multilingual
size_categories:
- 1K<n<10K
source_datasets:
- original
task_categories:
- translation
task_ids: []
pretty_name: OpusBooks
dataset_info:
- config_name: ca-de
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - ca
        - de
  splits:
  - name: train
    num_bytes: 899553
    num_examples: 4445
  download_size: 609128
  dataset_size: 899553
- config_name: ca-en
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - ca
        - en
  splits:
  - name: train
    num_bytes: 863162
    num_examples: 4605
  download_size: 585612
  dataset_size: 863162
- config_name: ca-hu
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - ca
        - hu
  splits:
  - name: train
    num_bytes: 886150
    num_examples: 4463
  download_size: 608827
  dataset_size: 886150
- config_name: ca-nl
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - ca
        - nl
  splits:
  - name: train
    num_bytes: 884811
    num_examples: 4329
  download_size: 594793
  dataset_size: 884811
- config_name: de-en
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - de
        - en
  splits:
  - name: train
    num_bytes: 13738975
    num_examples: 51467
  download_size: 8797832
  dataset_size: 13738975
- config_name: de-eo
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - de
        - eo
  splits:
  - name: train
    num_bytes: 398873
    num_examples: 1363
  download_size: 253509
  dataset_size: 398873
- config_name: de-es
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - de
        - es
  splits:
  - name: train
    num_bytes: 7592451
    num_examples: 27526
  download_size: 4841017
  dataset_size: 7592451
- config_name: de-fr
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - de
        - fr
  splits:
  - name: train
    num_bytes: 9544351
    num_examples: 34916
  download_size: 6164101
  dataset_size: 9544351
- config_name: de-hu
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - de
        - hu
  splits:
  - name: train
    num_bytes: 13514971
    num_examples: 51780
  download_size: 8814744
  dataset_size: 13514971
- config_name: de-it
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - de
        - it
  splits:
  - name: train
    num_bytes: 7759984
    num_examples: 27381
  download_size: 4901036
  dataset_size: 7759984
- config_name: de-nl
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - de
        - nl
  splits:
  - name: train
    num_bytes: 3561740
    num_examples: 15622
  download_size: 2290868
  dataset_size: 3561740
- config_name: de-pt
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - de
        - pt
  splits:
  - name: train
    num_bytes: 317143
    num_examples: 1102
  download_size: 197768
  dataset_size: 317143
- config_name: de-ru
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - de
        - ru
  splits:
  - name: train
    num_bytes: 5764649
    num_examples: 17373
  download_size: 3255537
  dataset_size: 5764649
- config_name: el-en
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - el
        - en
  splits:
  - name: train
    num_bytes: 552567
    num_examples: 1285
  download_size: 310863
  dataset_size: 552567
- config_name: el-es
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - el
        - es
  splits:
  - name: train
    num_bytes: 527979
    num_examples: 1096
  download_size: 298827
  dataset_size: 527979
- config_name: el-fr
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - el
        - fr
  splits:
  - name: train
    num_bytes: 539921
    num_examples: 1237
  download_size: 303181
  dataset_size: 539921
- config_name: el-hu
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - el
        - hu
  splits:
  - name: train
    num_bytes: 546278
    num_examples: 1090
  download_size: 313292
  dataset_size: 546278
- config_name: en-eo
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - en
        - eo
  splits:
  - name: train
    num_bytes: 386219
    num_examples: 1562
  download_size: 246715
  dataset_size: 386219
- config_name: en-es
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - en
        - es
  splits:
  - name: train
    num_bytes: 25291663
    num_examples: 93470
  download_size: 16080303
  dataset_size: 25291663
- config_name: en-fi
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - en
        - fi
  splits:
  - name: train
    num_bytes: 715027
    num_examples: 3645
  download_size: 467851
  dataset_size: 715027
- config_name: en-fr
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - en
        - fr
  splits:
  - name: train
    num_bytes: 32997043
    num_examples: 127085
  download_size: 20985324
  dataset_size: 32997043
- config_name: en-hu
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - en
        - hu
  splits:
  - name: train
    num_bytes: 35256766
    num_examples: 137151
  download_size: 23065198
  dataset_size: 35256766
- config_name: en-it
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - en
        - it
  splits:
  - name: train
    num_bytes: 8993755
    num_examples: 32332
  download_size: 5726189
  dataset_size: 8993755
- config_name: en-nl
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - en
        - nl
  splits:
  - name: train
    num_bytes: 10277990
    num_examples: 38652
  download_size: 6443323
  dataset_size: 10277990
- config_name: en-no
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - en
        - 'no'
  splits:
  - name: train
    num_bytes: 661966
    num_examples: 3499
  download_size: 429631
  dataset_size: 661966
- config_name: en-pl
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - en
        - pl
  splits:
  - name: train
    num_bytes: 583079
    num_examples: 2831
  download_size: 389337
  dataset_size: 583079
- config_name: en-pt
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - en
        - pt
  splits:
  - name: train
    num_bytes: 309677
    num_examples: 1404
  download_size: 191493
  dataset_size: 309677
- config_name: en-ru
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - en
        - ru
  splits:
  - name: train
    num_bytes: 5190856
    num_examples: 17496
  download_size: 2922360
  dataset_size: 5190856
- config_name: en-sv
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - en
        - sv
  splits:
  - name: train
    num_bytes: 790773
    num_examples: 3095
  download_size: 516328
  dataset_size: 790773
- config_name: eo-es
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - eo
        - es
  splits:
  - name: train
    num_bytes: 409579
    num_examples: 1677
  download_size: 265543
  dataset_size: 409579
- config_name: eo-fr
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - eo
        - fr
  splits:
  - name: train
    num_bytes: 412987
    num_examples: 1588
  download_size: 261689
  dataset_size: 412987
- config_name: eo-hu
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - eo
        - hu
  splits:
  - name: train
    num_bytes: 389100
    num_examples: 1636
  download_size: 258229
  dataset_size: 389100
- config_name: eo-it
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - eo
        - it
  splits:
  - name: train
    num_bytes: 387594
    num_examples: 1453
  download_size: 248748
  dataset_size: 387594
- config_name: eo-pt
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - eo
        - pt
  splits:
  - name: train
    num_bytes: 311067
    num_examples: 1259
  download_size: 197021
  dataset_size: 311067
- config_name: es-fi
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - es
        - fi
  splits:
  - name: train
    num_bytes: 710450
    num_examples: 3344
  download_size: 467281
  dataset_size: 710450
- config_name: es-fr
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - es
        - fr
  splits:
  - name: train
    num_bytes: 14382126
    num_examples: 56319
  download_size: 9164030
  dataset_size: 14382126
- config_name: es-hu
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - es
        - hu
  splits:
  - name: train
    num_bytes: 19373967
    num_examples: 78800
  download_size: 12691292
  dataset_size: 19373967
- config_name: es-it
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - es
        - it
  splits:
  - name: train
    num_bytes: 7837667
    num_examples: 28868
  download_size: 5026914
  dataset_size: 7837667
- config_name: es-nl
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - es
        - nl
  splits:
  - name: train
    num_bytes: 9062341
    num_examples: 32247
  download_size: 5661890
  dataset_size: 9062341
- config_name: es-no
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - es
        - 'no'
  splits:
  - name: train
    num_bytes: 729113
    num_examples: 3585
  download_size: 473525
  dataset_size: 729113
- config_name: es-pt
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - es
        - pt
  splits:
  - name: train
    num_bytes: 326872
    num_examples: 1327
  download_size: 204399
  dataset_size: 326872
- config_name: es-ru
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - es
        - ru
  splits:
  - name: train
    num_bytes: 5281106
    num_examples: 16793
  download_size: 2995191
  dataset_size: 5281106
- config_name: fi-fr
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - fi
        - fr
  splits:
  - name: train
    num_bytes: 746085
    num_examples: 3537
  download_size: 486904
  dataset_size: 746085
- config_name: fi-hu
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - fi
        - hu
  splits:
  - name: train
    num_bytes: 746602
    num_examples: 3504
  download_size: 509394
  dataset_size: 746602
- config_name: fi-no
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - fi
        - 'no'
  splits:
  - name: train
    num_bytes: 691169
    num_examples: 3414
  download_size: 449501
  dataset_size: 691169
- config_name: fi-pl
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - fi
        - pl
  splits:
  - name: train
    num_bytes: 613779
    num_examples: 2814
  download_size: 410258
  dataset_size: 613779
- config_name: fr-hu
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - fr
        - hu
  splits:
  - name: train
    num_bytes: 22483025
    num_examples: 89337
  download_size: 14689840
  dataset_size: 22483025
- config_name: fr-it
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - fr
        - it
  splits:
  - name: train
    num_bytes: 4752147
    num_examples: 14692
  download_size: 3040617
  dataset_size: 4752147
- config_name: fr-nl
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - fr
        - nl
  splits:
  - name: train
    num_bytes: 10408088
    num_examples: 40017
  download_size: 6528881
  dataset_size: 10408088
- config_name: fr-no
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - fr
        - 'no'
  splits:
  - name: train
    num_bytes: 692774
    num_examples: 3449
  download_size: 449136
  dataset_size: 692774
- config_name: fr-pl
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - fr
        - pl
  splits:
  - name: train
    num_bytes: 614236
    num_examples: 2825
  download_size: 408295
  dataset_size: 614236
- config_name: fr-pt
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - fr
        - pt
  splits:
  - name: train
    num_bytes: 324604
    num_examples: 1263
  download_size: 198700
  dataset_size: 324604
- config_name: fr-ru
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - fr
        - ru
  splits:
  - name: train
    num_bytes: 2474198
    num_examples: 8197
  download_size: 1425660
  dataset_size: 2474198
- config_name: fr-sv
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - fr
        - sv
  splits:
  - name: train
    num_bytes: 833541
    num_examples: 3002
  download_size: 545599
  dataset_size: 833541
- config_name: hu-it
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - hu
        - it
  splits:
  - name: train
    num_bytes: 8445537
    num_examples: 30949
  download_size: 5477452
  dataset_size: 8445537
- config_name: hu-nl
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - hu
        - nl
  splits:
  - name: train
    num_bytes: 10814113
    num_examples: 43428
  download_size: 6985092
  dataset_size: 10814113
- config_name: hu-no
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - hu
        - 'no'
  splits:
  - name: train
    num_bytes: 695485
    num_examples: 3410
  download_size: 465904
  dataset_size: 695485
- config_name: hu-pl
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - hu
        - pl
  splits:
  - name: train
    num_bytes: 616149
    num_examples: 2859
  download_size: 425988
  dataset_size: 616149
- config_name: hu-pt
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - hu
        - pt
  splits:
  - name: train
    num_bytes: 302960
    num_examples: 1184
  download_size: 193053
  dataset_size: 302960
- config_name: hu-ru
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - hu
        - ru
  splits:
  - name: train
    num_bytes: 7818652
    num_examples: 26127
  download_size: 4528613
  dataset_size: 7818652
- config_name: it-nl
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - it
        - nl
  splits:
  - name: train
    num_bytes: 1328293
    num_examples: 2359
  download_size: 824780
  dataset_size: 1328293
- config_name: it-pt
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - it
        - pt
  splits:
  - name: train
    num_bytes: 301416
    num_examples: 1163
  download_size: 190005
  dataset_size: 301416
- config_name: it-ru
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - it
        - ru
  splits:
  - name: train
    num_bytes: 5316928
    num_examples: 17906
  download_size: 2997871
  dataset_size: 5316928
- config_name: it-sv
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - it
        - sv
  splits:
  - name: train
    num_bytes: 811401
    num_examples: 2998
  download_size: 527303
  dataset_size: 811401
configs:
- config_name: ca-de
  data_files:
  - split: train
    path: ca-de/train-*
- config_name: ca-en
  data_files:
  - split: train
    path: ca-en/train-*
- config_name: ca-hu
  data_files:
  - split: train
    path: ca-hu/train-*
- config_name: ca-nl
  data_files:
  - split: train
    path: ca-nl/train-*
- config_name: de-en
  data_files:
  - split: train
    path: de-en/train-*
- config_name: de-eo
  data_files:
  - split: train
    path: de-eo/train-*
- config_name: de-es
  data_files:
  - split: train
    path: de-es/train-*
- config_name: de-fr
  data_files:
  - split: train
    path: de-fr/train-*
- config_name: de-hu
  data_files:
  - split: train
    path: de-hu/train-*
- config_name: de-it
  data_files:
  - split: train
    path: de-it/train-*
- config_name: de-nl
  data_files:
  - split: train
    path: de-nl/train-*
- config_name: de-pt
  data_files:
  - split: train
    path: de-pt/train-*
- config_name: de-ru
  data_files:
  - split: train
    path: de-ru/train-*
- config_name: el-en
  data_files:
  - split: train
    path: el-en/train-*
- config_name: el-es
  data_files:
  - split: train
    path: el-es/train-*
- config_name: el-fr
  data_files:
  - split: train
    path: el-fr/train-*
- config_name: el-hu
  data_files:
  - split: train
    path: el-hu/train-*
- config_name: en-eo
  data_files:
  - split: train
    path: en-eo/train-*
- config_name: en-es
  data_files:
  - split: train
    path: en-es/train-*
- config_name: en-fi
  data_files:
  - split: train
    path: en-fi/train-*
- config_name: en-fr
  data_files:
  - split: train
    path: en-fr/train-*
- config_name: en-hu
  data_files:
  - split: train
    path: en-hu/train-*
- config_name: en-it
  data_files:
  - split: train
    path: en-it/train-*
- config_name: en-nl
  data_files:
  - split: train
    path: en-nl/train-*
- config_name: en-no
  data_files:
  - split: train
    path: en-no/train-*
- config_name: en-pl
  data_files:
  - split: train
    path: en-pl/train-*
- config_name: en-pt
  data_files:
  - split: train
    path: en-pt/train-*
- config_name: en-ru
  data_files:
  - split: train
    path: en-ru/train-*
- config_name: en-sv
  data_files:
  - split: train
    path: en-sv/train-*
- config_name: eo-es
  data_files:
  - split: train
    path: eo-es/train-*
- config_name: eo-fr
  data_files:
  - split: train
    path: eo-fr/train-*
- config_name: eo-hu
  data_files:
  - split: train
    path: eo-hu/train-*
- config_name: eo-it
  data_files:
  - split: train
    path: eo-it/train-*
- config_name: eo-pt
  data_files:
  - split: train
    path: eo-pt/train-*
- config_name: es-fi
  data_files:
  - split: train
    path: es-fi/train-*
- config_name: es-fr
  data_files:
  - split: train
    path: es-fr/train-*
- config_name: es-hu
  data_files:
  - split: train
    path: es-hu/train-*
- config_name: es-it
  data_files:
  - split: train
    path: es-it/train-*
- config_name: es-nl
  data_files:
  - split: train
    path: es-nl/train-*
- config_name: es-no
  data_files:
  - split: train
    path: es-no/train-*
- config_name: es-pt
  data_files:
  - split: train
    path: es-pt/train-*
- config_name: es-ru
  data_files:
  - split: train
    path: es-ru/train-*
- config_name: fi-fr
  data_files:
  - split: train
    path: fi-fr/train-*
- config_name: fi-hu
  data_files:
  - split: train
    path: fi-hu/train-*
- config_name: fi-no
  data_files:
  - split: train
    path: fi-no/train-*
- config_name: fi-pl
  data_files:
  - split: train
    path: fi-pl/train-*
- config_name: fr-hu
  data_files:
  - split: train
    path: fr-hu/train-*
- config_name: fr-it
  data_files:
  - split: train
    path: fr-it/train-*
- config_name: fr-nl
  data_files:
  - split: train
    path: fr-nl/train-*
- config_name: fr-no
  data_files:
  - split: train
    path: fr-no/train-*
- config_name: fr-pl
  data_files:
  - split: train
    path: fr-pl/train-*
- config_name: fr-pt
  data_files:
  - split: train
    path: fr-pt/train-*
- config_name: fr-ru
  data_files:
  - split: train
    path: fr-ru/train-*
- config_name: fr-sv
  data_files:
  - split: train
    path: fr-sv/train-*
- config_name: hu-it
  data_files:
  - split: train
    path: hu-it/train-*
- config_name: hu-nl
  data_files:
  - split: train
    path: hu-nl/train-*
- config_name: hu-no
  data_files:
  - split: train
    path: hu-no/train-*
- config_name: hu-pl
  data_files:
  - split: train
    path: hu-pl/train-*
- config_name: hu-pt
  data_files:
  - split: train
    path: hu-pt/train-*
- config_name: hu-ru
  data_files:
  - split: train
    path: hu-ru/train-*
- config_name: it-nl
  data_files:
  - split: train
    path: it-nl/train-*
- config_name: it-pt
  data_files:
  - split: train
    path: it-pt/train-*
- config_name: it-ru
  data_files:
  - split: train
    path: it-ru/train-*
- config_name: it-sv
  data_files:
  - split: train
    path: it-sv/train-*
---

# Dataset Card for OPUS Books

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://opus.nlpl.eu/Books/corpus/version/Books
- **Repository:** [More Information Needed]
- **Paper:** https://aclanthology.org/L12-1246/
- **Leaderboard:** [More Information Needed]
- **Point of Contact:** [More Information Needed]

### Dataset Summary

This is a collection of copyright free books aligned by Andras Farkas, which are available from http://www.farkastranslations.com/bilingual_books.php
Note that the texts are rather dated due to copyright issues and that some of them are manually reviewed (check the meta-data at the top of the corpus files in XML). The source is multilingually aligned, which is available from http://www.farkastranslations.com/bilingual_books.php.
In OPUS, the alignment is formally bilingual but the multilingual alignment can be recovered from the XCES sentence alignment files. Note also that the alignment units from the original source may include multi-sentence paragraphs, which are split and sentence-aligned in OPUS.
All texts are freely available for personal, educational and research use. Commercial use (e.g. reselling as parallel books) and mass redistribution without explicit permission are not granted. Please acknowledge the source when using the data!

Books's Numbers:
- Languages: 16
- Bitexts: 64
- Number of files: 158
- Number of tokens: 19.50M
- Sentence fragments: 0.91M

### Supported Tasks and Leaderboards

Translation.

### Languages

The languages in the dataset are:
- ca
- de
- el
- en
- eo
- es
- fi
- fr
- hu
- it
- nl
- no
- pl
- pt
- ru
- sv

## Dataset Structure

### Data Instances

[More Information Needed]

### Data Fields

[More Information Needed]

### Data Splits

[More Information Needed]

## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

[More Information Needed]

#### Initial Data Collection and Normalization

[More Information Needed]

#### Who are the source language producers?

[More Information Needed]

### Annotations

[More Information Needed]

#### Annotation process

[More Information Needed]

#### Who are the annotators?

[More Information Needed]

### Personal and Sensitive Information

[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

[More Information Needed]

### Licensing Information

All texts are freely available for personal, educational and research use. Commercial use (e.g. reselling as parallel books) and mass redistribution without explicit permission are not granted.

### Citation Information

Please acknowledge the source when using the data.

Please cite the following article if you use any part of the OPUS corpus in your own work:
```bibtex
@inproceedings{tiedemann-2012-parallel,
    title = "Parallel Data, Tools and Interfaces in {OPUS}",
    author = {Tiedemann, J{\"o}rg},
    editor = "Calzolari, Nicoletta  and
      Choukri, Khalid  and
      Declerck, Thierry  and
      Do{\u{g}}an, Mehmet U{\u{g}}ur  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Eighth International Conference on Language Resources and Evaluation ({LREC}'12)",
    month = may,
    year = "2012",
    address = "Istanbul, Turkey",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf",
    pages = "2214--2218",
}
```

### Contributions

Thanks to [@abhishekkrthakur](https://github.com/abhishekkrthakur) for adding this dataset.