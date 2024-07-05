rm(list=ls())
setwd("~/projects/infolocality/")
library(tidyverse)
library(latex2exp)
library(ggplot2)
library(patchwork)

LOG2 = log(2)

BADS = c(
  "CMUDict",
  "CMUDict10000",
  "CMUDict20000",
  "english_sample",
  "FakeAyacuchoQuechua",
  "FakeN3AyacuchoQuechua",
  "FakeSouthernBritishEnglish",
  "FakeN3SouthernBritishEnglish",
  "FakeN3Turkish",
  "english",
  "SouthernBritishEnglishMonosyllables"
)
  

d = read_csv("output/wolex_analysis_20200309.csv") %>% 
  mutate(lang=str_replace(lang, ".Parsed.*", ""),
         real=factor(real, levels=c("real", "manner", "cv", "even_odd", "shuffled"))) %>%
  filter(!(lang %in% BADS)) %>%
  group_by(lang, real) %>%
    mutate(E=max(H_M_lower_bound)) %>%
    ungroup() %>%
  mutate(lang=if_else(lang=="SouthernBritishEnglish", "English", lang))

INTERESTING_LANGS = c(
  "English", # English
  "Turkish", # vowel harmony
  "Georgian", # consonant heavy
  "Zulu", # clicks
  "Hebrew", # nonconcatenative morphology 
  "Khmer" # tonal
)

df = d %>% filter(lang %in% INTERESTING_LANGS)

stats = d %>%
  group_by(lang, real) %>%
    summarize(E=max(H_M_lower_bound)) %>%
    ungroup() %>%
  mutate(type=if_else(real == "real", "real", "shuffled")) %>%
  mutate(p_value=str_c(type, " E=", format(E/LOG2, digits=2, nsmall=2), " bits"))

plots = df %>% 
  inner_join(stats) %>%
  filter(real %in% c("real", "manner")) %>%
  split(., .$lang) %>%
  lapply(function(data) {
    the_lang = unique(data$lang)
    the_plot = ggplot(data, aes(x=t+1, y=h_t/LOG2, color=p_value)) + 
      theme_classic() + 
      geom_line() + 
      scale_color_manual(values=c("black", "darkred")) +
      labs(x=TeX("$n$-gram Order"), y=TeX("$n$-gram Entropy Rate (bits)"), color="") +
      theme(legend.position=c(.67, .99),
            legend.background = element_rect(fill = "transparent")) +
      xlim(1,7)  +
      ggtitle(the_lang)
    the_plot
  })

wrap_plots(plots)

ggsave("plots/wolex_figures.pdf", height=4, width=9)

d %>% 
  inner_join(stats) %>%
  filter(real %in% c("real", "manner", "shuffled")) %>%
  mutate(real=factor(real, levels=c("manner", "shuffled", "real"))) %>%
  ggplot(aes(x=t+1, y=h_t/LOG2, color=real)) + 
      theme_classic() + 
      geom_line() + 
      facet_wrap(~lang, scale="free_y") +
      scale_color_manual(values=c("red", "blue", "black")) +
      labs(x="n-gram Order", y=TeX("n-gram Entropy Rate (bits)"), color="") +
      theme(legend.position="bottom") +
      xlim(0, 7)
      
ggsave("plots/all_wolex_figures.pdf", height=9, width=11)

stats %>%
  filter(real %in% c("real", "shuffled", "manner")) %>%
  select(lang, real, E) %>%
  mutate(E=E/LOG2) %>%
  spread(real, E) %>%
  print(n=100)
