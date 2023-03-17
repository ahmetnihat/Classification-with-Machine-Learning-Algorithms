from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


def find_missing_cols(dataFrame):
    """
    Eksik veriye sahip kolonları bulmak için oluşturduğumuz bir fonksiyondur.
    Kolonları liste şeklinde döndürür.
    """
    return [col for col in dataFrame.columns if dataFrame[col].isnull().any()]


def calc_object_cols(dataFrame, nunique = 0):
    """
    Kategorik verilere sahip kolonları bulmak için oluşturduğumuz bir fonksiyondur.
    Kolonları liste şeklinde döndürür.
    
    - nunique = en az kaç tane benzersiz bilgiye sahip kolonları almamız gerektiğini
    filtreleyen değişkenimizdir.
    
    - object_cols = kategorik verilere sahip kolonların isimlerini listeye ekler.
    
    - object_nunique = kategorik verilere sahip kolonların kaç tane benzersiz veriye
    sahip olduğunu verir.
    
    - d = Hangi kolonun kaç tane benzersiz veriye sahip olduğunu kolon ismi ile
    birlikte dictionary olarak verir.
    """
    
    object_cols = [cname for cname in dataFrame.columns if dataFrame[cname].dtype == "object" and dataFrame[cname].nunique() > nunique]
    
    object_nunique = list(map(lambda col: dataFrame[col].nunique(), object_cols))
    
    d = dict(zip(object_cols, object_nunique))
    # sözlüğümüzü sıralayarak vermek için kullanıyoruz.
    sorted(d.items(), key=lambda x: x[1])
    
    return object_cols, d


def create_pred_data(label, y_valid, preds, disp = None):
    
    """
    Tahminlerimizin doğruluğunu inceleyebilmek için oluşturduğumuzbir fonksiyondur.
    
    label = verilerimizin etiketidir.
    
    y_valid = target doğrulama verilerimizdir.
    
    preds = modelimizin verdiği tahminlerdir.
    
    Fonksiyonun aldığı tüm değişken türleri list'dir.
    
    Fonksiyon:
    
    digital_cm = sayısal confusion matrix oluşturur.
    percentile_cm = yüzdesel confusion matrix oluşturur.
    
    pred_percent = yüzdesel cm üzerinden doğru tahmin oranlarını çeker.
    
    Sum = sayısal cm üzerinden hangi sınıfın kaç tane veriye sahip olduğunu toplar.
    
    true = scm üzerinden her sınıf için doğru tahmin edilen verilerin sayısını listeler.
    
    false = sum - true
    
    most_frec = her sınıf için en fazla tahmin edilen sınıfı listeler. Yanlış sınıflandırmada en fazla 
    tekrar edilen sınıfa bir yanlılık olduğu gözlemlenebilir.
    """
    
    digital_cm = confusion_matrix(y_valid, preds)
    percentile_cm = confusion_matrix(y_valid, preds, normalize="true")
    
    pred_percent = [percentile_cm[i][i] for i in range(len(label))]
    Sum = [digital_cm[i].sum() for i in range(len(label))]
    true = [digital_cm[i][i] for i in range(len(label))]
    false = [Sum[i] - true[i] for i in range(len(label))]
    most_frec = [label[np.where(digital_cm[i] == max(digital_cm[i]))[0][0]] for i in range(len(label))]
    
    pred_df_list = list(zip(label, pred_percent, Sum, true, false, most_frec))

    pred_df = pd.DataFrame(pred_df_list, columns=["label", "TruePredsPercent", "Sum", "true", "false", "most_frec"])
    
    number_of_classes = pred_df["TruePredsPercent"].shape[0]
    sum_true_percentile = pred_df["TruePredsPercent"].sum()
    print(f" Toplam Doğruluk: % {(sum_true_percentile / number_of_classes) * 100}")
    
    if disp == None:
        pass
    elif disp == 0:
        print(ConfusionMatrixDisplay(confusion_matrix=percentile_cm, display_labels=[i for i in label]).plot())
    else:
        print(ConfusionMatrixDisplay(confusion_matrix=digital_cm, display_labels=[i for i in label]).plot())
        
    return pred_df


def get_score(n_estimators, X, y):
    """
    Return the average MAE over 3 CV folds of random forest model.
    
    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    
    # Replace this body with your own code
    my_pipeline = Pipeline(steps=[
        ('model', RandomForestRegressor(n_estimators, random_state=0))
    ])
    
    scores = -1 * cross_val_score(my_pipeline, X, y, cv=3, scoring='neg_mean_absolute_error')
    
    return scores.mean()


def model_fit_and_predict(model, y_label, X_train, y_train, X_valid, y_valid):

    """
    Model seçimi için kullanılır.
    karmaşıklık matrisi çizdirir ve tahminlerimizin üzerinde analiz yapabileceğimiz bir veri seti verir.
    """
    
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    
    return create_pred_data(y_label, y_valid, preds, 1)



def fast_model_select(model_list, X, y):
    """
    Liste şeklinde kıyaslama yapacağımız modellerimizi ve tüm verilerimizi alarak 3 çapraz doğrulama oluşturup
    modellerin doğruluklarını döndürür.
    """
    
    for i, model in enumerate(model_list):
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=3)
        print(f"Model {i} Accuracy: %{scores.mean()*100}")


print("DataMacpy Setup")