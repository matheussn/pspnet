#!/bin/bash

python new_main.py -dir new_pspnet_resnet_50_SGD -dataset Aug_NRR -log-interval 72 -epochs 50

sleep 10

python new_main.py -dir new_pspnet_resnet_50_ADAM -dataset Aug_NRR -log-interval 72 -epochs 50 -opt Adam

sleep 10

python new_main.py -dir new_pspnet_resnest_50_SGD -dataset Aug_NRR -log-interval 72 -epochs 50 -model pspnet_resnest.py

sleep 10

python new_main.py -dir new_pspnet_resnest_50_ADAM -dataset Aug_NRR -log-interval 72 -epochs 50 -model pspnet_resnest.py -opt Adam

sleep 10

python new_main.py -dir new_pspnet_unet_50_SGD -dataset Aug_NRR -log-interval 72 -epochs 50 -model pspnet_unet.py

sleep 10

python new_main.py -dir new_pspnet_unet_50_ADAM -dataset Aug_NRR -log-interval 72 -epochs 50 -model pspnet_unet.py -opt Adam

sleep 10

python new_main.py -dir new_pspnet_resnet_100_SGD -dataset Aug_NRR -log-interval 72 -epochs 100

sleep 10

python new_main.py -dir new_pspnet_resnet_100_ADAM -dataset Aug_NRR -log-interval 72 -epochs 100 -opt Adam

sleep 10

python new_main.py -dir new_pspnet_resnest_100_SGD -dataset Aug_NRR -log-interval 72 -epochs 100 -model pspnet_resnest.py

sleep 10

python new_main.py -dir new_pspnet_resnest_100_ADAM -dataset Aug_NRR -log-interval 72 -epochs 100 -model pspnet_resnest.py -opt Adam

sleep 10

python new_main.py -dir new_pspnet_unet_100_SGD -dataset Aug_NRR -log-interval 72 -epochs 100 -model pspnet_unet.py

sleep 10

python new_main.py -dir new_pspnet_unet_100_ADAM -dataset Aug_NRR -log-interval 72 -epochs 100 -model pspnet_unet.py -opt Adam

sleep 10

python new_main.py -dir new_pspnet_resnet_150_SGD -dataset Aug_NRR -log-interval 72 -epochs 150

sleep 10

python new_main.py -dir new_pspnet_resnet_150_ADAM -dataset Aug_NRR -log-interval 72 -epochs 150 -opt Adam

sleep 10

python new_main.py -dir new_pspnet_resnest_150_SGD -dataset Aug_NRR -log-interval 72 -epochs 150 -model pspnet_resnest.py

sleep 10

python new_main.py -dir new_pspnet_resnest_150_ADAM -dataset Aug_NRR -log-interval 72 -epochs 150 -model pspnet_resnest.py -opt Adam

sleep 10

python new_main.py -dir new_pspnet_unet_150_SGD -dataset Aug_NRR -log-interval 72 -epochs 150 -model pspnet_unet.py

sleep 10

python new_main.py -dir new_pspnet_unet_150_ADAM -dataset Aug_NRR -log-interval 72 -epochs 150 -model pspnet_unet.py -opt Adam

