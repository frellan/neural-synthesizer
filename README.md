# neural-synthesizer

```
python modular_train.py \
--dataset mnist \
--model simple \
--n_parts 2 \
--loss xe \
--lr1 .1 \
--lr2 .1 \
--activation relu \
--optimizer sgd \
--weight_decay1 .0002 \
--weight_decay2 .0002 \
--seed 5 \
--print_freq 1 \
--n_epochs1 70 \
--n_epochs2 70 \
--hidden_objective srs_upper_tri_alignment \
--loglevel info \
--n_classes 10 \
--augment_data False \
--in_channels 1 \
--batch_size 128 \
--n_val 5000 \
--max_trainset_size 55000
```
