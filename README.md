# neural-synthesizer

```
python morph_train.py \
--dataset mnist \
--loss xe \
--activation relu \
--optimizer sgd \
--print_freq 1 \
--hidden_objective srs_upper_tri_alignment \
--loglevel info \
--n_classes 10 \
--augment_data False \
--in_channels 1 \
--batch_size 128 \
--n_val 5000 \
--max_trainset_size 55000
```
