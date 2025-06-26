library(dplyr)

#wczytujemy dane do ramki danych
diabetes <- read.csv("C:\\Users\\Dorota\\Desktop\\Wstęp do analizy danych\\Dataset of Diabetes .csv", stringsAsFactors = FALSE)

#sprawdzamy poprawność wczytanych danych
str(diabetes)

#METODA k-NN

#zaczynamy od przygotowania danych do analizy
#kodujemy binarnie zmienną Gender i wykluczamy zmienne ID oraz No_Pation z analizy
diabetes <- diabetes %>% 
  mutate(Gender= ifelse(Gender == "F", 1, 0)) %>% # Female ma wartość 1
  select(-ID,-No_Pation)

str(diabetes)
#ID oraz No_Pation zostały usunięte

#w naszym projekcie zmienną przewidywaną jest zmienna CLASS
#kodujemy ją jako czynnik oraz nadajemy jej wartościom odpowiednie etykiety
diabetes$CLASS <- factor(diabetes$CLASS, levels = c("N","Y", "P"), labels = c("Non-diabetic", "Diabetic", "Prediabetic"))

#sprawdzamy, czy istnieją wiersze, w których występują braki danych
sum(!complete.cases(diabetes))
#funkcja zwraca liczbę takich wierszy - jest ich 5, usuwamy je
diabetes<- diabetes %>% filter(complete.cases(.))

sum(!complete.cases(diabetes))
#brak wierszy z brakami danych

#podsumowanie dla zmiennych numerycznych
summary(diabetes[1:11])

#sprawdzamy liczebność poszczególnych klas dla czynnika
table(diabetes$CLASS)

#przechodzimy do normalizacji danych
#tworzymy funkcję, która normalizuje wszystkie współrzędne wektora
normalize<- function(x) {
  return ((x-min(x))/(max(x)-min(x)))
}

#normalizujemy dane
diabetes_n <- as.data.frame(lapply(diabetes[1:11], normalize))

#sprawdzamy, czy normalizacja zadziałała poprawnie
summary(diabetes_n)
#sukces - wszystkie zmienne numeryczne przyjmują wartości z przedziału [0,1]

#przechodzimy do tworzenia zbioru uczącego i testowego

#ustawiamy seed generatora losowego
set.seed(123)

#losujemy unikalne indeksy
all_indices <- sample(nrow(diabetes_n), 995)
train_indices <- all_indices[1:800]
test_indices <- all_indices[801:995]

diabetes_train <- diabetes_n[train_indices, ]
diabetes_test <- diabetes_n[test_indices, ]
diabetes_train_labels <- diabetes[train_indices, 12]
diabetes_test_labels <- diabetes[test_indices, 12]

#będziemy używać funkcji knn() z pakietu class
library(class)

#dane uczące składają się z 800 przykładów - w pierwszej kolejności spróbujmy k równe przybliżonej wartości pierwiastka z 800
sqrt(800)
diabetes_test_pred <- knn(train=diabetes_train, test=diabetes_test, cl=diabetes_train_labels, k=29)  

table(diabetes_test_pred)

#użyjemy funkcję CrossTable() w stworzenia tabeli krzyżowej przedstawiającej,
#w jakim stopniu przewidziane klasy pokrywają się z prawdziwymi klasami
library(gmodels)

CrossTable(x= diabetes_test_labels, y=diabetes_test_pred, prop.chisq = FALSE) 

#testujemy inne wartości parametru k
diabetes_test_pred1 <- knn(train=diabetes_train, test=diabetes_test, cl=diabetes_train_labels, k=27)  
table(diabetes_test_pred1)
CrossTable(x= diabetes_test_labels, y=diabetes_test_pred1, prop.chisq = FALSE) 

diabetes_test_pred2 <- knn(train=diabetes_train, test=diabetes_test, cl=diabetes_train_labels, k=3)  
table(diabetes_test_pred2)
CrossTable(x= diabetes_test_labels, y=diabetes_test_pred2, prop.chisq = FALSE)

#dopracowujemy model
#tym razem stosujemy standaryzację z-score
#używamy funkcji scale() do przeprowadzenia standaryzacji z-score na danych wyjściowych
diabetes_z <- as.data.frame(scale(diabetes[-12]))
summary(diabetes_z)

#przechodzimy do tworzenia zbiorów uczącego i testowego
set.seed(123)

#losujemy unikalne indeksy
all_indices_z <- sample(nrow(diabetes_z), 995)
train_indices_z <- all_indices_z[1:800]
test_indices_z <- all_indices_z[801:995]

diabetes_train_z <- diabetes_z[train_indices_z, ]
diabetes_test_z <- diabetes_z[test_indices_z, ]
diabetes_train_labels_z <- diabetes[train_indices_z, 12]
diabetes_test_labels_z <- diabetes[test_indices_z, 12]

#testujemy różne wartości k i tworzymy tabele krzyżowe
diabetes_test_pred_z <- knn(train=diabetes_train_z, test=diabetes_test_z, cl=diabetes_train_labels_z, k=15)
CrossTable(x= diabetes_test_labels_z, y=diabetes_test_pred_z, prop.chisq = FALSE) 

diabetes_test_pred_z1 <- knn(train=diabetes_train_z, test=diabetes_test_z, cl=diabetes_train_labels_z, k=3)
CrossTable(x= diabetes_test_labels_z, y=diabetes_test_pred_z1, prop.chisq = FALSE) 


#NAIWNA KLASYFIKACJA BAYESOWSKA

#wciąż czynnikiem jest zmienna CLASS
str(diabetes$CLASS)
table(diabetes$CLASS)

#sprawdzamy czy odsetek cukrzyków jest podobny w zbiorze uczącym i testowym
prop.table(table(diabetes_train_labels)) #84,5% to cukrzycy
prop.table(table(diabetes_test_labels)) #84,1% to cukrzycy

#będziemy używać funkcji naiveBayes() z pakietu e1071
library(e1071)

diabetes_nb <- naiveBayes(diabetes_train, diabetes_train_labels)
diabetes_test_pred_nb <- predict(diabetes_nb, diabetes_test)

#tworzymy tabelę krzyżową
CrossTable(diabetes_test_pred_nb, diabetes_test_labels, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, prop.c = FALSE, dnn = c("Predicted", "Actual"))

#dopracowujemy model
#zastosujemy wygładzenie Laplace'a z parametrem laplace=1

#budowa drugiego modelu
diabetes_nb2 <- naiveBayes(diabetes_train, diabetes_train_labels, laplace=1)

#predykcja z użyciem drugiego modelu
diabetes_test_pred2 <- predict(diabetes_nb2, diabetes_test)

CrossTable(diabetes_test_pred2, diabetes_test_labels, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, prop.c = FALSE, dnn = c("Predicted", "Actual"))  
#takie same wyniki -> przetestujmy inne wartości parametru laplace

#wygładzenie Laplace'a z parametrem laplace=0.5
diabetes_nb3 <- naiveBayes(diabetes_train, diabetes_train_labels, laplace=0.5)
diabetes_test_pred3 <- predict(diabetes_nb3, diabetes_test)
CrossTable(diabetes_test_pred3, diabetes_test_labels, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, prop.c = FALSE, dnn = c("Predicted", "Actual"))  

#wygładzenie Laplace'a z parametrem laplace=2
diabetes_nb4 <- naiveBayes(diabetes_train, diabetes_train_labels, laplace=2)
diabetes_test_pred4 <- predict(diabetes_nb4, diabetes_test)
CrossTable(diabetes_test_pred4, diabetes_test_labels, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, prop.c = FALSE, dnn = c("Predicted", "Actual"))  

#za każdym razem otrzymujemy taką samą tabelę krzyżową
#oznacza to, że wszystkie potrzebne kombinacje cech i klas pojawiły się choć raz w zbiorze treninowym


#DRZEWA DECYZYJNE

#dzielimy dane na uczące i testowe
set.seed(123)
train_sample <- sample(995,800)
str(train_sample)

diabetes_train1 <- diabetes[train_sample,]
diabetes_test1 <- diabetes[-train_sample,]

#sprawdzamy prpoprcje klas w utworzonych zbiorach
prop.table(table(diabetes_train1$CLASS))
prop.table(table(diabetes_test1$CLASS))

#budujemy model
#użyjemy funkcji C5.0() z pakietu C50
library(C50)

#budujemy drzewo decyzyjne
diabetes_model <- C5.0(diabetes_train1[-12], diabetes_train1$CLASS)

#wyświetlamy proste informacje o zbudowanym drzewie
diabetes_model

#struktura drzewa
summary(diabetes_model)

#predykcje dla danych testowych
diabetes_pred1 <- predict(diabetes_model, diabetes_test1)

#tabela krzyżowa
CrossTable(diabetes_test1$CLASS, diabetes_pred1, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, prop.c = FALSE, dnn = c("Predicted", "Actual"))

#dopracowujemy model
#używamy metody AdaBoost z parametrem trials=10
diabetes_boost <- C5.0(diabetes_train1[-12], diabetes_train1$CLASS, trials=10)

#stworzony obiekt
diabetes_boost

#drzewa wchodzące w skład zbudowanego modelu
summary(diabetes_boost)

#stosujemy model AdaBoost do danych testowych
diabetes_pred_boost <- predict(diabetes_boost, diabetes_test1)

#tabela krzyżowa
CrossTable(diabetes_test1$CLASS, diabetes_pred_boost, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, prop.c = FALSE, dnn = c("Predicted", "Actual"))

#dopracowujemy model
#zastosujemy metodę przypisania kar różnym rodzajom błędów
#kary podamy w postaci macierzy kosztów
#tutaj mamy do czynienia z trzema klasami, więc wymiar macierzy kosztów będzie wynosił 3x3

#tworzymy macierz kosztów
matrix_dimensions<- list(c("Non-diabetic", "Diabetic", "Prediabetic"), c("Non-diabetic", "Diabetic", "Prediabetic"))
names(matrix_dimensions) <- c("Predicted", "Actual")

#sprawdzenie
matrix_dimensions

#nadajemy kary za różne rodzaje błędów
error_cost <- matrix(c(0,0,3,0,2,0,1,0,0), nrow=3, dimnames = matrix_dimensions)

#sprawdzenie poprawności macierzy kosztów
error_cost

#uwzględniamy macierz kosztów w drzewie decyzyjnym
diabetes_cost <- C5.0(diabetes_train1[-12], diabetes_train1$CLASS, costs=error_cost)
diabetes_cost_pred<- predict(diabetes_cost, diabetes_test1)

#tworzymy tabelę krzyżową
CrossTable(diabetes_test1$CLASS, diabetes_cost_pred, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, prop.c = FALSE, dnn = c("Predicted", "Actual"))






