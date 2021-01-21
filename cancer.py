from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
cancer = load_breast_cancer()
model = SVC()
df_cancer = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
df_target = pd.DataFrame(cancer['target'], columns = ['Cancer'])
y = np.ravel(df_target)
poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC( degree=3, coef0=1, C=5))
])
X_train, X_test, y_train, y_test = train_test_split(df_cancer, np.ravel(df_target), test_size=0.20, random_state=101)
model.fit(X_train, y_train)
poly_kernel_svm_clf.fit(df_cancer, y)
pm = model.predict(X_test)
pm1 = accuracy_score(y_test, pm)
while True:
    r = float(input('raio: '))
    tex = float(input('textura: '))
    peri = float(input('perímetro: '))
    are = float(input('área: '))
    sua = float(input('suavidade: '))
    comp = float(input('compactação: '))
    conv = float(input('concavidade: '))
    pconv = float(input('pontos côncavos: '))
    sim = float(input('simetria: '))
    dfrac = float(input('dimensão fractal: '))
    rE = float(input('raioE: '))
    texE = float(input('texturaE: '))
    periE = float(input('perimetroE: '))
    areE = float(input('áreaE: '))
    suaE = float(input('suavidadeE: '))
    compE = float(input('compactaçãoE: '))
    convE = float(input('concavidadeE: '))
    pconvE = float(input('pontos côncavosE: '))
    simE = float(input('simetriaE: '))
    dfracE = float(input('dimensão fractal E: '))
    rW = float(input('raioW: '))
    texW = float(input('texturaW: '))
    periW = float(input('perimetroW: '))
    areW = float(input('áreaW: '))
    suaW = float(input('suavidadeW: '))
    compW = float(input('compactacãoW: '))
    convW = float(input('concavidadeW: '))
    pconvW = float(input('pontos côncavos W: '))
    simW = float(input('simetriaW: '))
    dfracW = float(input('dimensão fractal W: '))
    sv = np.array([r, tex, peri, are, sua, comp, conv, pconv, sim, dfrac, rE, texE, periE, areE, suaE, compE, convE, pconvE, simE, dfracE, rW, texW, periW, areW, suaW, compW, convW, pconvW, simW, dfracW])
    sv = sv.reshape(1, -1)
    pr = model.predict(sv)
    if pr == 1:
        pr = 'maligna'
    else:
        pr = 'benigna'
    print('/n')
    print(pr)
    print(classification_report(y_test, pm))
    print('/n')
    print(confusion_matrix(y_test, pm))
    pergunta = input('deseja continuar?(s/n): ') 
    if pergunta == 'n':
        break       
