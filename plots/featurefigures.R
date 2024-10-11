rm(list=ls())
setwd("~/projects/infolocality")
library(tidyverse)
library(ggrepel)

LOG2 = log(2)

d_word = read_csv("output/word_feature_corrs.csv") %>%
  mutate(label=str_c(str_replace(i, ".mean", ""), " × ", str_replace(j, ".mean", ""))) 

d_morph = read_csv("output/morpheme_feature_corrs.csv") %>%
  mutate(label=str_c(str_replace(i, ".mean", ""), " × ", str_replace(j, ".mean", ""))) 

d_morph %>%
  mutate(label=reorder(label, mi)) %>%
  mutate(within=if_else(within, "Within Morpheme", "Across Morphemes")) %>%
  ggplot(aes(x=label, y=mi/LOG2, fill=within, color=within, label=label)) +
    geom_bar(stat="identity") +
    geom_text(angle=90, size=1.75, hjust=-.1) +
    theme_classic() +
    labs(x="Feature Pair", y="Mutual Information (bits)") +
    theme(axis.ticks.x=element_blank(),
          axis.text.x=element_blank(),
          legend.title=element_blank(),
          legend.position=c(.2, .8))

ggsave("plots/lancaster_mi_numberfeature.pdf", width=7, height=2)

THIN = 2

lucky_labels = d_word %>%
  arrange(mi) %>%
  mutate(rank=1:n()) %>%
  mutate(lucky=rank %% THIN == THIN - 1) %>%
  filter(lucky) %>%
  select(label) %>%
  inner_join(select(d_word, label, mi, within)) %>%
  mutate(within=if_else(within, "Within Word", "Across Words"))
  
d_word %>%
  mutate(label=reorder(label, mi)) %>%
  mutate(within=if_else(within, "Within Word", "Across Words")) %>%
  ggplot(aes(x=label, y=mi/LOG2, fill=within, color=within)) +
    geom_bar(stat="identity") +
    geom_text(aes(label=label), data=lucky_labels, size=1.75, hjust=-.1, angle=90) +
    theme_classic() +
    labs(x="Feature Pair", y="Mutual Information (bits)") +
    theme(axis.ticks.x=element_blank(),
          axis.text.x=element_blank(),
          legend.title=element_blank(),
          legend.position=c(.2, .8))

ggsave("plots/lancaster_mi_verbobject.pdf", width=7, height=2)




    
    
  
  