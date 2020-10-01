# chorus
# INFO:__main__:{'loss': 0.038295578211545944,
#                'mae': 0.14439217746257782}
# INFO:__main__:y_cont granular MAE = [0.1099736  0.23523717 0.2215074  0.14365272 0.12369812 0.03228392]
# INFO:__main__:mean mse = 0.0082
# INFO:__main__:std mse = 0.0082
# INFO:__main__:mean mae = 0.0508
# INFO:__main__:std mae = 0.0318

# compressor
# INFO:__main__:{'loss': 0.0009621087228879333,
#                'mae': 0.020053954795002937}
# INFO:__main__:y_cont granular MAE = [0.0174068  0.02829562 0.01445946]
# INFO:__main__:mean mse = 0.0023
# INFO:__main__:std mse = 0.0021
# INFO:__main__:mean mae = 0.0239
# INFO:__main__:std mae = 0.0153

# distortion
# INFO: __main__:{'loss': 0.3983064591884613,
#                 'dist_mode_loss': 0.36372750997543335,
#                 'cont_output_loss': 0.03457893803715706,
#                 'dist_mode_acc': 0.8669999837875366,
#                 'cont_output_mae': 0.13383513689041138}
# INFO:__main__:n_perfect_pred = 2
# INFO:__main__:perfect pred = 0.20%
# INFO:__main__:mean mse = 0.0185
# INFO:__main__:std mse = 0.0339
# INFO:__main__:mean mae = 0.0728
# INFO:__main__:std mae = 0.0787

# eq
# INFO:__main__:{'loss': 0.883407473564148,
#                'eq_typl_loss': 0.5597400069236755,
#                'eq_typh_loss': 0.2613202631473541,
#                'cont_output_loss': 0.0623471662402153,
#                'eq_typl_acc': 0.7680000066757202,
#                'eq_typh_acc': 0.890999972820282,
#                'cont_output_mae': 0.20215460658073425}
# INFO:__main__:y_cont granular MAE = [0.1690348  0.1691671  0.23900047 0.23175344 0.2012385  0.20273289]
# INFO:__main__:mean mse = 0.0352
# INFO:__main__:std mse = 0.0492
# INFO:__main__:mean mae = 0.1248
# INFO:__main__:std mae = 0.0893

# filter
# INFO:__main__:{'loss': 2.0681264400482178,
#                'fx_fil_type_loss': 1.997994065284729,
#                'cont_output_loss': 0.0701325386762619,
#                'fx_fil_type_acc': 0.3919999897480011,
#                'cont_output_mae': 0.22187009453773499}
# INFO:__main__:y_cont granular MAE = [0.17660359 0.23106153 0.18933657 0.2514241  0.2609247 ]
# INFO:__main__:mean mse = 0.0592
# INFO:__main__:std mse = 0.0664
# INFO:__main__:mean mae = 0.1671
# INFO:__main__:std mae = 0.0955

# flanger
# INFO:__main__:{'loss': 0.022245261818170547,
#                'mae': 0.09270791709423065}
# INFO:__main__:y_cont granular MAE = [0.04631894 0.06270943 0.03522167 0.22658174]
# INFO:__main__:mean mse = 0.0218
# INFO:__main__:std mse = 0.0144
# INFO:__main__:mean mae = 0.1010
# INFO:__main__:std mae = 0.0349

# phaser
# INFO:__main__:{'loss': 0.025490181520581245,
#                'mae': 0.10815902054309845}
# INFO:__main__:y_cont granular MAE = [0.1029989  0.09811173 0.05666324 0.03905379 0.24396719]
# INFO:__main__:mean mse = 0.0247
# INFO:__main__:std mse = 0.0220
# INFO:__main__:mean mae = 0.1083
# INFO:__main__:std mae = 0.0521

# reverb-hall
# INFO:__main__:{'loss': 0.03149664029479027,
#                'mae': 0.13025474548339844}
# INFO:__main__:y_cont granular MAE = [0.09304024 0.17627536 0.09998553 0.18617088 0.06083341 0.16522278]
# INFO:__main__:mean mse = 0.0339
# INFO:__main__:std mse = 0.0076
# INFO:__main__:mean mae = 0.1657
# INFO:__main__:std mae = 0.0255


# training_eq_l
# saw 56k baseline cnn

# eq
# loss: 0.1923 - eq_typl_loss: 0.1514 - cont_output_loss: 0.0409 - eq_typl_acc: 0.9402 - cont_output_mae: 0.1566
# val_loss: 0.2436 - val_eq_typl_loss: 0.2071 - val_cont_output_loss: 0.0365 - val_eq_typl_acc: 0.9248 - val_cont_output_mae: 0.1447
# multi
# loss: 0.1730 - eq_typl_loss: 0.1343 - cont_output_loss: 0.0387 - eq_typl_acc: 0.9467 - cont_output_mae: 0.1510
# val_loss: 0.2712 - val_eq_typl_loss: 0.2363 - val_cont_output_loss: 0.0349 - val_eq_typl_acc: 0.9089 - val_cont_output_mae: 0.1400

# compressor
# loss: 0.0049 - mae: 0.0533 - val_loss: 0.0039 - val_mae: 0.0425
# multi
# loss: 0.0050 - mae: 0.0536 - val_loss: 0.0040 - val_mae: 0.0439

# distortion
# loss: 0.4207 - dist_mode_loss: 0.3757 - cont_output_loss: 0.0451 - dist_mode_acc: 0.8445 - cont_output_mae: 0.1678
# val_loss: 0.5214 - val_dist_mode_loss: 0.4812 - val_cont_output_loss: 0.0402 - val_dist_mode_acc: 0.8132 - val_cont_output_mae: 0.1536
# multi
# loss: 0.4153 - dist_mode_loss: 0.3701 - cont_output_loss: 0.0451 - dist_mode_acc: 0.8440 - cont_output_mae: 0.1674
# val_loss: 0.5112 - val_dist_mode_loss: 0.4700 - val_cont_output_loss: 0.0411 - val_dist_mode_acc: 0.8209 - val_cont_output_mae: 0.1568


# basic shapes

# distortion

# baseline cnn 2x
# loss: 0.4964 - dist_mode_loss: 0.4445 - cont_output_loss: 0.0519 - dist_mode_acc: 0.8090 - cont_output_mae: 0.1834
# val_loss: 0.5108 - val_dist_mode_loss: 0.4669 - val_cont_output_loss: 0.0439 - val_dist_mode_acc: 0.7986 - val_cont_output_mae: 0.1660
# baseline cnn
# loss: 0.4849 - dist_mode_loss: 0.4343 - cont_output_loss: 0.0506 - dist_mode_acc: 0.8115 - cont_output_mae: 0.1806
# val_loss: 0.5252 - val_dist_mode_loss: 0.4815 - val_cont_output_loss: 0.0437 - val_dist_mode_acc: 0.8005 - val_cont_output_mae: 0.1652
# exposure cnn
# loss: 0.6227 - dist_mode_loss: 0.5424 - cont_output_loss: 0.0803 - dist_mode_acc: 0.7761 - cont_output_mae: 0.2309
# val_loss: 0.6168 - val_dist_mode_loss: 0.5515 - val_cont_output_loss: 0.0653 - val_dist_mode_acc: 0.7797 - val_cont_output_mae: 0.2114
# baseline lstm
# loss: 0.5199 - dist_mode_loss: 0.4714 - cont_output_loss: 0.0485 - dist_mode_acc: 0.8018 - cont_output_mae: 0.1743
# val_loss: 0.6044 - val_dist_mode_loss: 0.5589 - val_cont_output_loss: 0.0455 - val_dist_mode_acc: 0.7726 - val_cont_output_mae: 0.1666

# phaser

# baseline cnn 2x
# loss: 0.0176 - mae: 0.0992 - val_loss: 0.0175 - val_mae: 0.0876
# loss: 0.0205 - mae: 0.1081 - val_loss: 0.0211 - val_mae: 0.1011  (x, x)
# baseline cnn
# loss: 0.0202 - mae: 0.1037 - val_loss: 0.0195 - val_mae: 0.0963

# eq

# baseline cnn 2x
# loss: 0.2039 - eq_typl_loss: 0.1600 - cont_output_loss: 0.0439 - eq_typl_acc: 0.9341 - cont_output_mae: 0.1629
# val_loss: 0.2483 - val_eq_typl_loss: 0.2119 - val_cont_output_loss: 0.0365 - val_eq_typl_acc: 0.9115 - val_cont_output_mae: 0.1440
# loss: 0.4000 - eq_typl_loss: 0.3351 - cont_output_loss: 0.0649 - eq_typl_acc: 0.8398 - cont_output_mae: 0.2089   (x, x)
# val_loss: 0.4419 - val_eq_typl_loss: 0.3806 - val_cont_output_loss: 0.0613 - val_eq_typl_acc: 0.8099 - val_cont_output_mae: 0.2026  (x, x)

# compressor

# baseline cnn 2x
# loss: 0.0120 - mae: 0.0813 - val_loss: 0.0102 - val_mae: 0.0670
# loss: 0.0211 - mae: 0.1103 - val_loss: 0.0225 - val_mae: 0.1065  (x, x)
# baseline cnn
# loss: 0.0118 - mae: 0.0773 - val_loss: 0.0110 - val_mae: 0.0692