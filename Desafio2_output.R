## DESAFIO 2 - PRECIFICANDO UM IMOVEL COM REGRESSAO LINEAR

# como vimos na analise exploratoria, 
# existem muitos outliers na variavel Area
# dessa forma agrupamos os imoveis com base em sua area e preco
# assim podemos construir algoritmos especificos para cada cenario
# uma vez que outliers prejudicam o poder de predicao do modelo de regressao linear


# bibliotecas para manipulacao de dados
library(tidyverse)

# biliotecas graficas
library(plotly)
library(ggplot2)

# bibliotecas para tratamento de variaveis dummy
library(fastDummies)

# bibliotecas para avaliacao do modelo
library(jtools)
library(nortest)
library(car)
library(olsrr)
library(randomForest)
library(rpart.plot)


## ANALISE GRUPO 1

# importando os dados 
dados <- read.csv('grupo1.csv')

#Plotando o Preco em funcaoo da Area para validar a dispersao dos dados
ggplotly(
  ggplot(dados, aes(x = Area, y = Preco)) +
    geom_point(color = "orange") +
    geom_smooth(aes(color = "Fitted Values"),
                method = "lm", 
                level = 0.95) +
    labs(x = "Area",
         y = "Preco", 
         title = 'Grupo 1') +
    scale_color_manual("Legenda:",
                       values = "darkorchid") +
    theme_bw()
)

# Ajustando variaveis qualitativas
dados$Zona <- as.factor(dados$Zona)
dados$Qualidade <- as.factor(dados$Qualidade)
dados$QualidadeAquecimento <- as.factor(dados$QualidadeAquecimento)

# Estatisticas descritivas sobre o grupo 1
# importando ressaltar que o modelo treinado sobre esses dados
# somente podera realizar predicoes dentro do intervalo de cada variavel
# sem poder realizar extrapolacao dos dados

summary(dados)
# Area entre 6000 e 12735
# AnoConstrucao entre 1940 e 2007
# Banheiros entre 1 e 2 
# Quartost1 entre 1 e 3
# Quartost2 entre 2 e 4
# Comodos entre 4 e 8
# Lareiras entre 0 e 2
# Garagem entre 0 e 3

# Transformando variaveis dummy
dados_dummy <- dummy_columns(.data = dados,
                             select_columns = c('Zona',
                                                'Qualidade',
                                                'QualidadeAquecimento'),
                             remove_most_frequent_dummy = T,
                             remove_selected_columns = T)


# Estimando um modelo de regressao linear
# aplicando o procedimento stepwise para excluir
# variaveis que nao se mostram significantes 
# na presenca das demais
modelo1 <- lm(formula = Preco ~ . , data = dados_dummy)
modelo1 <- step(modelo1, k = qchisq(df = 1, p = 0.05, lower.tail = F))

# avaliando o modelo
summ(modelo1, digits = 6)
# obs. todos os betas ficaram com p-valor < 0.05
# o que siginifica que explicam bem o comportamento de y = Preco
# r2 = 0.31, relativamente baixo, devido a dispersao dos dados

# avaliando se os termos de erro tendem a normalidade
sf.test(modelo1$residuals)
# obs. como p-valor < 0.05, isso mostra que os termos de erro
# nao tendem a normalidade sendo necessario transformar Y

# Plotando os termos de erro e os fitted.values
# obs. linearidade dos termos de erro
# caso a linha azul adote um corpotamento curvo, 
# seria interessante trabalhar com modelo nao linear
residualPlot(modelo1)

# vamos realizar a transformacao da variavel Y e rodar o modelo novamente
# a transformacao de box cox tenta aumentar a tendencia dos residuos a normalidade

lambda <- powerTransform(dados_dummy$Preco)
dados_dummy$PrecoBC <- (((dados_dummy$Preco^lambda$lambda)-1)/lambda$lambda)
modelo1 <- lm(formula = PrecoBC ~ . - Preco, data = dados_dummy)
modelo1 <- step(modelo1, k = qchisq(p=0.05, df=1, lower.tail = F))

# Avaliando o modelo
summ(modelo1)
# obs. todos os betas ficaram com p-valor < 0.05
# o que siginifica que explicam bem o comportamento de y = PrecoBC
# r2 = 0.33, aumentou em relacao ao primeiro modelo
# porem continua baixo, devido a dispersao dos dados

# Novamente realizando o teste de normalidade dos termos de erro
sf.test(modelo1$residuals)
# obs. o p-valor nao ficou acima de 0.05 como esperado
# ficou 0.03, porem um valor maior que o obtido anteriormente
# o que indica que a transformacao de Y foi favoravel 

# plotando os fitted values pelos residuos para analisar a linearidade
residualPlot(modelo1)
# obs. e possivel perceber que a linha azul ficou menos curva que a anterior


# Diagnostico de heterocedasticidade
# Se o p-valor < 0.05 entao os termos de erro possui correlacao
# com alguma variavel explicativa
# o que indica que faltam variaveis relevantes no dataset
ols_test_breusch_pagan(modelo1)
# obs. resultado de 0.10, logo o modelo passou no teste

# Adicionando o valor previsto pelo modelo no dataset e salvando a base
dados$PrecoPrevisto <- (((modelo1$fitted.values*lambda$lambda) + 1)) ^ (1 / lambda$lambda)
write.csv(dados, file = 'output/grupo1.csv')
view(dados)



##############################################################
## ANALISE GRUPO 2

# importando os dados 
dados <- read.csv('grupo2.csv')

#Plotando o Preco em funcao da Area para validar a dispersao dos dados
ggplotly(
  ggplot(dados, aes(x = Area, y = Preco)) +
    geom_point(color = "orange") +
    geom_smooth(aes(color = "Fitted Values"),
                method = "lm", 
                level = 0.95) +
    labs(x = "Area",
         y = "Preco", 
         title = 'Grupo 1') +
    scale_color_manual("Legenda:",
                       values = "darkorchid") +
    theme_bw()
)

# Ajustando variaveis qualitativas
dados$Zona <- as.factor(dados$Zona)
dados$Qualidade <- as.factor(dados$Qualidade)
dados$QualidadeAquecimento <- as.factor(dados$QualidadeAquecimento)

# Estatisticas descritivas sobre o grupo 1
# importando ressaltar que o modelo treinado sobre esses dados
# somente podera realizar predicoes dentro do intervalo de cada variavel
# sem poder realizar extrapolacao dos dados

summary(dados)
# Area entre 7024 e 12384
# AnoConstrucao entre 1940 e 2009
# Banheiros entre 1 e 2 
# Quartost1 entre 1 e 3
# Quartost2 entre 2 e 4
# Comodos entre 4 e 8
# Lareiras entre 0 e 2
# Garagem entre 1 e 3

# Transformando variáveis dummy
dados_dummy <- dummy_columns(.data = dados,
                             select_columns = c('Zona',
                                                'Qualidade',
                                                'QualidadeAquecimento'),
                             remove_most_frequent_dummy = T,
                             remove_selected_columns = T)


# Estimando um modelo de regressao linear
# aplicando o procedimento stepwise para excluir
# variaveis que nao se mostram significantes 
# na presenca das demais

modelo2 <- lm(formula = log(Preco) ~ . , data = dados_dummy)
modelo2 <- step(modelo2, k = qchisq(df = 1, p = 0.05, lower.tail = F))

# avaliando o modelo
summ(modelo2, digits = 6)
# obs. todos os betas ficaram com p-valor < 0.05
# o que siginifica que explicam bem o comportamento de y = Preco
# r2 = 0.52

# avaliando se os termos de erro tendem a normalidade
sf.test(modelo2$residuals)
# obs. como p-valor > 0.05, isso mostra que os termos de erro
# tendem a normalidade nao sendo necessario transformar Y por box cox

# Plotando os termos de erro e os fitted.values
# obs. linearidade dos termos de erro
# caso a linha azul adote um corpotamento curvo, 
# seria interessante trabalhar com modelo nao linear 
residualPlot(modelo2)
# Diagnostico de heterocedasticidade
# Se o p-valor < 0.05 entao os termos de erro possui correlacao
# com alguma variavel explicativa
# o que indica que faltam variaveis relevantes no dataset
ols_test_breusch_pagan(modelo2)
# obs. resultado de 0.45, logo o modelo passou no teste


# Adicionando o valor previsto pelo modelo no dataset e salvando a base
dados$PrecoPrevisto <- exp(modelo2$fitted.values)
write.csv(dados, file = 'output/grupo2.csv')
view(dados)




##############################################################
## ANALISE GRUPO 3

# importando os dados 
dados <- read.csv('grupo3.csv')

#Plotando o Preco em funcao da Area para validar a dispersao dos dados
ggplotly(
  ggplot(dados, aes(x = Area, y = Preco)) +
    geom_point(color = "orange") +
    geom_smooth(aes(color = "Fitted Values"),
                method = "lm", 
                level = 0.95) +
    labs(x = "Area",
         y = "Preco", 
         title = 'Grupo 1') +
    scale_color_manual("Legenda:",
                       values = "darkorchid") +
    theme_bw()
)

# Ajustando variaveis qualitativas
dados$Zona <- as.factor(dados$Zona)
dados$Qualidade <- as.factor(dados$Qualidade)
dados$QualidadeAquecimento <- as.factor(dados$QualidadeAquecimento)

# Estatisticas descritivas sobre o grupo 1
# importando ressaltar que o modelo treinado sobre esses dados
# somente podera realizar predicoes dentro do intervalo de cada variavel
# sem poder realizar extrapolacao dos dados

summary(dados)
# Area entre 1300 e 6000
# AnoConstrucao entre 1940 e 2007
# Banheiros entre 1 e 2 
# Quartost1 entre 1 e 3
# Quartost2 entre 2 e 4
# Comodos entre 3 e 7
# Lareiras entre 0 e 2
# Garagem entre 0 e 2

# Transformando variáveis dummy
dados_dummy <- dummy_columns(.data = dados,
                             select_columns = c('Zona',
                                                'Qualidade',
                                                'QualidadeAquecimento'),
                             remove_most_frequent_dummy = T,
                             remove_selected_columns = T)


# Estimando um modelo de regressao linear
# aplicando o procedimento stepwise para excluir
# variaveis que nao se mostram significantes 
# na presenca das demais

modelo3 <- lm(formula = Preco ~ . , data = dados_dummy)
modelo3 <- step(modelo3, k = qchisq(df = 1, p = 0.05, lower.tail = F))

# avaliando o modelo
summ(modelo3, digits = 6)
# obs. todos os betas ficaram com p-valor < 0.05
# o que siginifica que explicam bem o comportamento de y = Preco
# r2 = 0.84, valor legal, indica que o modelo esta conseguindo capturar
# o comportamento de Y

# avaliando se os termos de erro tendem a normalidade
sf.test(modelo3$residuals)
# obs. como p-valor > 0.05, isso mostra que os termos de erro
# ficou em 0.27
# tendem a normalidade nao sendo necessario transformar Y por box cox

# Plotando os termos de erro e os fitted.values
# obs. linearidade dos termos de erro
# caso a linha azul adote um corpotamento curvo, 
# seria interessante trabalhar com modelo nao linear 
residualPlot(modelo3)

# podemos observar que a linha azul tende a ser curva
# porem como passou no teste anteriormente vamos manter o modelo assim

# Diagnostico de heterocedasticidade
# Se o p-valor < 0.05 entao os termos de erro possui correlacao
# com alguma variavel explicativa
# o que indica que faltam variaveis relevantes no dataset
ols_test_breusch_pagan(modelo3)
# obs. resultado de 0.26, logo o modelo passou no teste


# Adicionando o valor previsto pelo modelo no dataset e salvando a base
dados$PrecoPrevisto <- modelo3$fitted.values
write.csv(dados, file = 'output/grupo3.csv')
view(dados)



##############################################################
## ANALISE GRUPO 4

# importando os dados 
dados <- read.csv('grupo4.csv')

#Plotando o Preco em funcao da Area para validar a dispersao dos dados
ggplotly(
  ggplot(dados, aes(x = Area, y = Preco)) +
    geom_point(color = "orange") +
    geom_smooth(aes(color = "Fitted Values"),
                method = "lm", 
                level = 0.95) +
    labs(x = "Area",
         y = "Preco", 
         title = 'Grupo 1') +
    scale_color_manual("Legenda:",
                       values = "darkorchid") +
    theme_bw()
)

# Ajustando variaveis qualitativas
dados$Zona <- as.factor(dados$Zona)
dados$Qualidade <- as.factor(dados$Qualidade)
dados$QualidadeAquecimento <- as.factor(dados$QualidadeAquecimento)

# Estatisticas descritivas sobre o grupo 1
# importando ressaltar que o modelo treinado sobre esses dados
# somente podera realizar predicoes dentro do intervalo de cada variavel
# sem poder realizar extrapolacao dos dados

summary(dados)
# Area entre 10920 e 19900
# AnoConstrucao entre 1941 e 2006
# Banheiros entre 1 e 2 
# Quartost1 entre 1 e 3
# Quartost2 entre 2 e 4
# Comodos entre 4 e 8
# Lareiras entre 0 e 2
# Garagem entre 1 e 3
# Transformando variáveis dummy
dados_dummy <- dummy_columns(.data = dados,
                             select_columns = c('Zona',
                                                'Qualidade',
                                                'QualidadeAquecimento'),
                             remove_most_frequent_dummy = T,
                             remove_selected_columns = T)


# Estimando um modelo de regressao linear
# aplicando o procedimento stepwise para excluir
# variaveis que nao se mostram significantes 
# na presenca das demais

modelo4 <- lm(formula = Preco ~ . , data = dados_dummy)
modelo4 <- step(modelo4, k = qchisq(df = 1, p = 0.05, lower.tail = F))

# avaliando o modelo
summ(modelo4, digits = 6)
# obs. todos os betas ficaram com p-valor < 0.05
# o que siginifica que explicam bem o comportamento de y = Preco
# r2 = 0.41, baixo, devido a dispersao dos dados

# avaliando se os termos de erro tendem a normalidade
sf.test(modelo4$residuals)
# obs. como p-valor < 0.05, isso mostra que os termos de erro
# nao tendem a normalidade sendo necessario transformar Y por box cox

# Plotando os termos de erro e os fitted.values
# obs. linearidade dos termos de erro
# caso a linha azul adote um corpotamento curvo, 
# seria interessante trabalhar com modelo nao linear 
residualPlot(modelo4)

# podemos observar que a linha azul tende a ser curva
# vamos realizar a transformacao da variavel Y e rodar o modelo novamente
# a transformacao de box cox tenta aumentar a tendencia dos residuos a normalidade

lambda <- powerTransform(dados_dummy$Preco)
dados_dummy$PrecoBC <- (((dados_dummy$Preco^lambda$lambda)-1)/lambda$lambda)
modelo4 <- lm(formula = PrecoBC ~ . - Preco, data = dados_dummy)
modelo4 <- step(modelo4, k = qchisq(p=0.05, df=1, lower.tail = F))

# Avaliando o modelo
summ(modelo4)
# obs. todos os betas ficaram com p-valor < 0.05
# o que siginifica que explicam bem o comportamento de y = PrecoBC
# r2 = 0.42, aumentou em relacao ao primeiro modelo
# porem continua baixo, devido a dispersao dos dados

# Novamente realizando o teste de normalidade dos termos de erro
sf.test(modelo4$residuals)
# obs. o p-valor ficou acima de 0.05 como esperado
# ficou 0.06
# o que indica que a transformacao de Y foi favorável 

# plotando os fitted values pelos residuos para analisar a linearidade
residualPlot(modelo4)
# obs. e possivel perceber que a linha azul ficou menos curva que a anterior
# esse modelo conseguira prever valores muito proximos de Y nos dados centrais
# enquanto os dados das extremidades (precos muito baixos ou muito altos do conjunto de dados)
# terao erros maiores

# Diagnostico de heterocedasticidade
# Se o p-valor < 0.05 entao os termos de erro possui correlacao
# com alguma variavel explicativa
# o que indica que faltam variaveis relevantes no dataset
ols_test_breusch_pagan(modelo4)
# obs. resultado de 0.25, logo o modelo passou no teste


# Adicionando o valor previsto pelo modelo no dataset e salvando a base
dados$PrecoPrevisto <- (((modelo4$fitted.values*lambda$lambda) + 1)) ^ (1 / lambda$lambda)
write.csv(dados, file = 'output/grupo4.csv')
view(dados)


##############################
# Conclusao: Nosso melhor modelo foi a regressao linear treinada com
# o conjunto de dados do grupo 3
# com r2 de 0.84
# seria o melhor modelo entre os 4
# porem precisamos considerar que as predicoes devem estar dentro do intervalo
# de treino das variaveis preditoras

###############################
# Vamos utilizar o conjunto de dados do grupo3 
# para rodar um modelo baseado em randomForest e outro em CART
# assim podemos comparar qual modelo se ajusta melhor
# um modelo glm, decisiontree ou ensemble

dados <- read.csv('grupo3.csv')
dados$Zona <- as.factor(dados$Zona)
dados$Qualidade <- as.factor(dados$Qualidade)
dados$QualidadeAquecimento <- as.factor(dados$QualidadeAquecimento)

# nao e necessario realizar o procedimento de variaveis dummy
# para modelos baseados em arvores

RF <- randomForest(formula = Preco ~ .,
                   data = dados,
                   importance =T,
                   mtry = 3)

DT <- rpart(formula = Preco ~ .,
            data = dados,
            control = rpart.control(minsplit = 10, maxdepth = 5))

rpart.plot(DT)

dados$PrecoRF <- predict(RF, newdata = dados)
dados$PrecoRG <- modelo3$fitted.values
dados$PrecoDT <- predict(DT, newdata = dados)

dados %>% 
  ggplot() +
  geom_line(aes(x = Preco, y = Preco, color = "Valores Reais")) +
  geom_smooth(aes(x = Preco, y = PrecoRF, color = "RandomForest"), se = FALSE) +
  geom_smooth(aes(x = Preco, y = PrecoRG, color = "RegressaoLinear"), se = FALSE) +
  geom_smooth(aes(x = Preco, y = PrecoDT, color = "ArvoreDecisao"), se = FALSE) +
  labs(x = NULL, y = NULL) +
  scale_color_manual("Legenda:", values = c("green", "orange", 
                                            "darkorchid", "black")) +
  theme_bw()

# Analisando o grafico podemos perceber que o modelo de arvore de decisao
# e randomforest tiveram resultados bem proximos 
# para valores menores randomforest se ajusta melhor
# enquanto para valores maiores arvore de decisao 
# regressao linear acaba ficando pior em relacao aos modelos de arvores
# vamos salvar a base de dados do grupo 3 com as previsoes dos 3 modelos
write.csv(dados, 'output/grupo3_decisionTree.csv')
view(dados)

# fim
