
# Teaching regression: A basic tutorial on how to build a regression model from start to finish
By Joe Ganser

### INTRODUCTION

In this notebook, I will be creating a simple work flow tutorial that allows beginner data science students to go through the process of building a regression model. This is for students who have completed most of my introduction to data science class and have basic familiarity with data science.

This notebook includes many key work flow processes involved in any data science project, such as:
* Exploring Data before Feature Engineering
* Cleaning Data & Feature Engineering
* Exploring Data after cleaning/feature engineering
* Feature Selection
* Building regression model
* Validating and testing a model
* Getting statistical inferences from the model

**Background info**

The data set we'll be working with is well known and studied. It describes real-estate sales on houses in AMES Iowa, and the target variable we're trying to predict is the price of a home. Lets load the data and begin exploring. A lot of [kaggle](https://www.kaggle.com) projects have been made from it, and you can find them [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

### EXPLORING DATA - before feature engineering
In any data science project, we always start of by getting a good glimpse of what kind of data we're looking at. This is called "exploratory data analysis" (EDA). There are a few things we need to look at at our first round of EDA:
1. The number of rows/columns.
2. The data types on each columns.
3. Get the min, max, mean values of each column
4. The number of missing values.

Lets start by loading the dataset, and observing it's dimensions. Our target variable `y` is the house price, and the rest of the variables (`X`) are our predictors. We want to load the file `train.csv` because it will train our models.


```python
#Solution
```

Now split the data set into a predictor table (`X`) and a target table `y`


```python
#Solution
```

Now look at the head of the predictor table, and print it's shape.


```python
#Code Your Solution Here
```


```python
#Don't alter this cell so you can see what your answer should look like.
```

    (1460, 80)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>



Now use the `.info()` method to find that datatypes on all of the columns (including the target!)


```python
#Code Your Solution Here
```


```python
#Don't alter this cell so you can see what your answer should look like.
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
    Id               1460 non-null int64
    MSSubClass       1460 non-null int64
    MSZoning         1460 non-null object
    LotFrontage      1201 non-null float64
    LotArea          1460 non-null int64
    Street           1460 non-null object
    Alley            91 non-null object
    LotShape         1460 non-null object
    LandContour      1460 non-null object
    Utilities        1460 non-null object
    LotConfig        1460 non-null object
    LandSlope        1460 non-null object
    Neighborhood     1460 non-null object
    Condition1       1460 non-null object
    Condition2       1460 non-null object
    BldgType         1460 non-null object
    HouseStyle       1460 non-null object
    OverallQual      1460 non-null int64
    OverallCond      1460 non-null int64
    YearBuilt        1460 non-null int64
    YearRemodAdd     1460 non-null int64
    RoofStyle        1460 non-null object
    RoofMatl         1460 non-null object
    Exterior1st      1460 non-null object
    Exterior2nd      1460 non-null object
    MasVnrType       1452 non-null object
    MasVnrArea       1452 non-null float64
    ExterQual        1460 non-null object
    ExterCond        1460 non-null object
    Foundation       1460 non-null object
    BsmtQual         1423 non-null object
    BsmtCond         1423 non-null object
    BsmtExposure     1422 non-null object
    BsmtFinType1     1423 non-null object
    BsmtFinSF1       1460 non-null int64
    BsmtFinType2     1422 non-null object
    BsmtFinSF2       1460 non-null int64
    BsmtUnfSF        1460 non-null int64
    TotalBsmtSF      1460 non-null int64
    Heating          1460 non-null object
    HeatingQC        1460 non-null object
    CentralAir       1460 non-null object
    Electrical       1459 non-null object
    1stFlrSF         1460 non-null int64
    2ndFlrSF         1460 non-null int64
    LowQualFinSF     1460 non-null int64
    GrLivArea        1460 non-null int64
    BsmtFullBath     1460 non-null int64
    BsmtHalfBath     1460 non-null int64
    FullBath         1460 non-null int64
    HalfBath         1460 non-null int64
    BedroomAbvGr     1460 non-null int64
    KitchenAbvGr     1460 non-null int64
    KitchenQual      1460 non-null object
    TotRmsAbvGrd     1460 non-null int64
    Functional       1460 non-null object
    Fireplaces       1460 non-null int64
    FireplaceQu      770 non-null object
    GarageType       1379 non-null object
    GarageYrBlt      1379 non-null float64
    GarageFinish     1379 non-null object
    GarageCars       1460 non-null int64
    GarageArea       1460 non-null int64
    GarageQual       1379 non-null object
    GarageCond       1379 non-null object
    PavedDrive       1460 non-null object
    WoodDeckSF       1460 non-null int64
    OpenPorchSF      1460 non-null int64
    EnclosedPorch    1460 non-null int64
    3SsnPorch        1460 non-null int64
    ScreenPorch      1460 non-null int64
    PoolArea         1460 non-null int64
    PoolQC           7 non-null object
    Fence            281 non-null object
    MiscFeature      54 non-null object
    MiscVal          1460 non-null int64
    MoSold           1460 non-null int64
    YrSold           1460 non-null int64
    SaleType         1460 non-null object
    SaleCondition    1460 non-null object
    SalePrice        1460 non-null int64
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB


Now use the `.describe()` method on the full data set to get the min,max,mean and other values for each column. Try adding in the `.transpose()` method to make it a little more readable.


```python
#Code your solution here.
```


```python
#Don't alter this cell so you can see what your answer should look like.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Id</th>
      <td>1460.0</td>
      <td>730.500000</td>
      <td>421.610009</td>
      <td>1.0</td>
      <td>365.75</td>
      <td>730.5</td>
      <td>1095.25</td>
      <td>1460.0</td>
    </tr>
    <tr>
      <th>MSSubClass</th>
      <td>1460.0</td>
      <td>56.897260</td>
      <td>42.300571</td>
      <td>20.0</td>
      <td>20.00</td>
      <td>50.0</td>
      <td>70.00</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>1201.0</td>
      <td>70.049958</td>
      <td>24.284752</td>
      <td>21.0</td>
      <td>59.00</td>
      <td>69.0</td>
      <td>80.00</td>
      <td>313.0</td>
    </tr>
    <tr>
      <th>LotArea</th>
      <td>1460.0</td>
      <td>10516.828082</td>
      <td>9981.264932</td>
      <td>1300.0</td>
      <td>7553.50</td>
      <td>9478.5</td>
      <td>11601.50</td>
      <td>215245.0</td>
    </tr>
    <tr>
      <th>OverallQual</th>
      <td>1460.0</td>
      <td>6.099315</td>
      <td>1.382997</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>OverallCond</th>
      <td>1460.0</td>
      <td>5.575342</td>
      <td>1.112799</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>5.0</td>
      <td>6.00</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>YearBuilt</th>
      <td>1460.0</td>
      <td>1971.267808</td>
      <td>30.202904</td>
      <td>1872.0</td>
      <td>1954.00</td>
      <td>1973.0</td>
      <td>2000.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>YearRemodAdd</th>
      <td>1460.0</td>
      <td>1984.865753</td>
      <td>20.645407</td>
      <td>1950.0</td>
      <td>1967.00</td>
      <td>1994.0</td>
      <td>2004.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>1452.0</td>
      <td>103.685262</td>
      <td>181.066207</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>166.00</td>
      <td>1600.0</td>
    </tr>
    <tr>
      <th>BsmtFinSF1</th>
      <td>1460.0</td>
      <td>443.639726</td>
      <td>456.098091</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>383.5</td>
      <td>712.25</td>
      <td>5644.0</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>1460.0</td>
      <td>46.549315</td>
      <td>161.319273</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1474.0</td>
    </tr>
    <tr>
      <th>BsmtUnfSF</th>
      <td>1460.0</td>
      <td>567.240411</td>
      <td>441.866955</td>
      <td>0.0</td>
      <td>223.00</td>
      <td>477.5</td>
      <td>808.00</td>
      <td>2336.0</td>
    </tr>
    <tr>
      <th>TotalBsmtSF</th>
      <td>1460.0</td>
      <td>1057.429452</td>
      <td>438.705324</td>
      <td>0.0</td>
      <td>795.75</td>
      <td>991.5</td>
      <td>1298.25</td>
      <td>6110.0</td>
    </tr>
    <tr>
      <th>1stFlrSF</th>
      <td>1460.0</td>
      <td>1162.626712</td>
      <td>386.587738</td>
      <td>334.0</td>
      <td>882.00</td>
      <td>1087.0</td>
      <td>1391.25</td>
      <td>4692.0</td>
    </tr>
    <tr>
      <th>2ndFlrSF</th>
      <td>1460.0</td>
      <td>346.992466</td>
      <td>436.528436</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>728.00</td>
      <td>2065.0</td>
    </tr>
    <tr>
      <th>LowQualFinSF</th>
      <td>1460.0</td>
      <td>5.844521</td>
      <td>48.623081</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>572.0</td>
    </tr>
    <tr>
      <th>GrLivArea</th>
      <td>1460.0</td>
      <td>1515.463699</td>
      <td>525.480383</td>
      <td>334.0</td>
      <td>1129.50</td>
      <td>1464.0</td>
      <td>1776.75</td>
      <td>5642.0</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>1460.0</td>
      <td>0.425342</td>
      <td>0.518911</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>BsmtHalfBath</th>
      <td>1460.0</td>
      <td>0.057534</td>
      <td>0.238753</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>FullBath</th>
      <td>1460.0</td>
      <td>1.565068</td>
      <td>0.550916</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>HalfBath</th>
      <td>1460.0</td>
      <td>0.382877</td>
      <td>0.502885</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>BedroomAbvGr</th>
      <td>1460.0</td>
      <td>2.866438</td>
      <td>0.815778</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>KitchenAbvGr</th>
      <td>1460.0</td>
      <td>1.046575</td>
      <td>0.220338</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>TotRmsAbvGrd</th>
      <td>1460.0</td>
      <td>6.517808</td>
      <td>1.625393</td>
      <td>2.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>Fireplaces</th>
      <td>1460.0</td>
      <td>0.613014</td>
      <td>0.644666</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>1379.0</td>
      <td>1978.506164</td>
      <td>24.689725</td>
      <td>1900.0</td>
      <td>1961.00</td>
      <td>1980.0</td>
      <td>2002.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>1460.0</td>
      <td>1.767123</td>
      <td>0.747315</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>1460.0</td>
      <td>472.980137</td>
      <td>213.804841</td>
      <td>0.0</td>
      <td>334.50</td>
      <td>480.0</td>
      <td>576.00</td>
      <td>1418.0</td>
    </tr>
    <tr>
      <th>WoodDeckSF</th>
      <td>1460.0</td>
      <td>94.244521</td>
      <td>125.338794</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>168.00</td>
      <td>857.0</td>
    </tr>
    <tr>
      <th>OpenPorchSF</th>
      <td>1460.0</td>
      <td>46.660274</td>
      <td>66.256028</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>25.0</td>
      <td>68.00</td>
      <td>547.0</td>
    </tr>
    <tr>
      <th>EnclosedPorch</th>
      <td>1460.0</td>
      <td>21.954110</td>
      <td>61.119149</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>552.0</td>
    </tr>
    <tr>
      <th>3SsnPorch</th>
      <td>1460.0</td>
      <td>3.409589</td>
      <td>29.317331</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>508.0</td>
    </tr>
    <tr>
      <th>ScreenPorch</th>
      <td>1460.0</td>
      <td>15.060959</td>
      <td>55.757415</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>480.0</td>
    </tr>
    <tr>
      <th>PoolArea</th>
      <td>1460.0</td>
      <td>2.758904</td>
      <td>40.177307</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>738.0</td>
    </tr>
    <tr>
      <th>MiscVal</th>
      <td>1460.0</td>
      <td>43.489041</td>
      <td>496.123024</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>15500.0</td>
    </tr>
    <tr>
      <th>MoSold</th>
      <td>1460.0</td>
      <td>6.321918</td>
      <td>2.703626</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>8.00</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>YrSold</th>
      <td>1460.0</td>
      <td>2007.815753</td>
      <td>1.328095</td>
      <td>2006.0</td>
      <td>2007.00</td>
      <td>2008.0</td>
      <td>2009.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>SalePrice</th>
      <td>1460.0</td>
      <td>180921.195890</td>
      <td>79442.502883</td>
      <td>34900.0</td>
      <td>129975.00</td>
      <td>163000.0</td>
      <td>214000.00</td>
      <td>755000.0</td>
    </tr>
  </tbody>
</table>
</div>



Now lets code up a script to create a bar chart that counts the number of missing values in each of the predictor columns. Intutively, we could suggest that a feature with a lot of missing values might not be important. This may or may not be true, but it's important to know what is and isn't there to begin with.


```python
#Code your solution here.
```


```python
#Don't alter this cell so you can see what your answer should look like.
```


![png](output_18_0.png)


### FEATURE ENGINEERING AND DATA CLEANING

Cleaning data and manipulating features is a major part of any project. In this part of a data science project we solve problems such as:

* Turning categorical (string) variables into numerical ones
* Dealing with missing values by changing them or removing them
* Dealing with outliers by changing them or removing them
* Make corrections to data types so machine learning algorithms can read them; e.g. removing a dollar sign to conver `'$4.51'` to `4.51` so it's a float
* Grouping columns of different data types to be dealt with differently

First, lets seperate the numerical predictors from the cateogircal (string) ones. Do this by going through the list of columns and selecting those with `.dtypes(column)=='object'` for categorical, and not equal for numerical.


```python
#Code Your solution here
```

Now lets look at the categorical features. Print the shape and display the head of the table.


```python
#Code your solution here
```


```python
#Don't alter this cell so you can see what your answer should look like.
```

    (1460, 43)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSZoning</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>...</th>
      <th>GarageType</th>
      <th>GarageFinish</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>...</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>...</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>...</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>...</td>
      <td>Detchd</td>
      <td>Unf</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Abnorml</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>...</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43 columns</p>
</div>



Now we have to convert this categorical data into numerical data. This is done using the pandas `.get_dummies(data)` function, which basically makes a new column for each categorical variable in a given column, and puts a value of 1 if that category is present for that row. The down side of doing this is the number of variables then blows up, which we will deal with later during feature selection.

Acting on the original table `df`, we get dummies of the categories, drop the categories from the table `df` and then concatenate the dummies to table `df`. This will lead us with only numbers in our dataframe, no strings.

If you did it right you should now have 290 columns.


```python
#Code your solution here
```


```python
#Don't alter this cell so you can see what your answer should look like.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Family</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 290 columns</p>
</div>



## FEATURE SELECTION 

Now that we've engineered our features, we should create a dataframe or series that shows the pearson correlation coefficient between the sale price and each feature. Get the histogram of this dataframe. Also print the standard deviation, mean, max and min values of the correlations.


```python
#Code your solution here
```


```python
#Don't alter this cell so you can see what your answer should look like.
```


![png](output_30_0.png)


    Max Correlation:  1.0
    Min Correlation:  -0.589043523409763
    Mean Correlation:  0.03910614218619199
    Standard Deviation:  0.2129964043689813 
    
    SalePrice                1.000000
    OverallQual              0.790982
    GrLivArea                0.708624
    GarageCars               0.640409
    GarageArea               0.623431
    TotalBsmtSF              0.613581
    1stFlrSF                 0.605852
    FullBath                 0.560664
    BsmtQual_Ex              0.553105
    TotRmsAbvGrd             0.533723
    YearBuilt                0.522897
    YearRemodAdd             0.507101
    KitchenQual_Ex           0.504094
    Foundation_PConc         0.497734
    GarageYrBlt              0.486362
    MasVnrArea               0.477493
    Fireplaces               0.466929
    ExterQual_Gd             0.452466
    ExterQual_Ex             0.451164
    BsmtFinType1_GLQ         0.434597
    HeatingQC_Ex             0.434543
    GarageFinish_Fin         0.419678
    Neighborhood_NridgHt     0.402149
    BsmtFinSF1               0.386420
    SaleType_New             0.357509
    SaleCondition_Partial    0.352060
    LotFrontage              0.351799
    FireplaceQu_Gd           0.339329
    GarageType_Attchd        0.335961
    MasVnrType_Stone         0.330476
                               ...   
    Fence_MnPrv             -0.140613
    Neighborhood_BrkSide    -0.143648
    SaleCondition_Normal    -0.153990
    KitchenQual_Fa          -0.157199
    Exterior1st_Wd Sdng     -0.158619
    Exterior2nd_Wd Sdng     -0.161800
    Exterior2nd_MetalSd     -0.162389
    HouseStyle_1.5Fin       -0.163466
    Neighborhood_IDOTRR     -0.164056
    Exterior1st_MetalSd     -0.167068
    Neighborhood_Edwards    -0.179949
    Neighborhood_NAmes      -0.188513
    Neighborhood_OldTown    -0.192189
    Electrical_FuseA        -0.193978
    Foundation_BrkTil       -0.204117
    PavedDrive_N            -0.212630
    RoofStyle_Gable         -0.224744
    SaleType_WD             -0.242598
    CentralAir_N            -0.251328
    BsmtExposure_No         -0.263600
    LotShape_Reg            -0.267672
    MSZoning_RM             -0.288065
    HeatingQC_TA            -0.312677
    Foundation_CBlock       -0.343263
    GarageType_Detchd       -0.354141
    MasVnrType_None         -0.374468
    GarageFinish_Unf        -0.410608
    BsmtQual_TA             -0.452394
    KitchenQual_TA          -0.519298
    ExterQual_TA            -0.589044
    Name: SalePrice, Length: 290, dtype: float64


So the if the correlation between the sale price and a feature is very small or close to zero, it's valid to say that such a feature has little to no influence on our price. The important features are going to be ones that have a significantly large enough absolute value of their correlation coefficient.

Thus we could specify a threshold of correlation that selects only features whose correlation with the SalePrice has an absolute value greater than that threshold. This could take a little bit of guess work, but we will try with a correlation greater than $\|\pm0.4\|$.

Using the series of correlation coefficients just made, select all the ones with an absolute value greater than 0.4, and select the features corresponding to the coefficients. You should have 26 features. Call these features the `key_features`.


```python
#Code your solution here.
```




    26



Now lets scatter plot each key feature compared to the sale price. This will allow us to detect things like skews in the data, which we can impute/transform in the following steps.


```python
#Code your solution here.
```


```python
#Don't alter this cell so you can see what your answer should look like.
```


![png](output_35_0.png)



![png](output_35_1.png)



![png](output_35_2.png)



![png](output_35_3.png)



![png](output_35_4.png)



![png](output_35_5.png)



![png](output_35_6.png)



![png](output_35_7.png)



![png](output_35_8.png)



![png](output_35_9.png)



![png](output_35_10.png)



![png](output_35_11.png)



![png](output_35_12.png)



![png](output_35_13.png)



![png](output_35_14.png)



![png](output_35_15.png)



![png](output_35_16.png)



![png](output_35_17.png)



![png](output_35_18.png)



![png](output_35_19.png)



![png](output_35_20.png)



![png](output_35_21.png)



![png](output_35_22.png)



![png](output_35_23.png)



![png](output_35_24.png)



![png](output_35_25.png)


Its very obvious that some of these features are very skewed. Moreover, the magnitude of some of the features are on significantly different scales. For example, the values for `GarageFinish_Unf` have a range of 0 and 1, but the feature `GarageYrBuilt` ranges from 1900 to 2010. Machine learning models work best with unskewed data which is all on a similar scale.

So what can we do? We can use a data-transformation technique that allows us to get all the data on the same scale, and (hopefully) removes the skew in the data.

Do this in several steps:
* For the missing values in the original data frame, impute the mean value into each column.
* From `sklearn.preprocessing`, use `StandardScaler` to transform our our columns into a standardized format


```python
#Code your solution here
```


```python
#Don't alter this cell so you can see what your answer should look like.
```

And what about our target variable, the `SalePrice`? Should it be transformed as well?
Using the `seaborn` function `.distplot(data,fit=norm)`, plot the sale price. Below it, plot the `stats.probplot(data,plot=plt)` as well and see what it looks like. Make you have imported all the right things.

This accomplishes two tasks:
* [Distplot](https://seaborn.pydata.org/generated/seaborn.distplot.html) will allow us to see how close our SalePrices are to a normal distribution, by creating a historgram.
* [Probplot](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.probplot.html) will allow us to see a Q-Q plot of our SalesPrices, generating a probability of sample data against the quantiles of a normal distribution.


```python
#Code your solution here
```


```python
#Don't alter this cell so you can see what your answer should look like.
```


![png](output_41_0.png)



![png](output_41_1.png)


Now do the same thing again, but transform the `SalePrice` data with the function `np.log1p(data)`.


```python
#Code your solution here
```


```python
#Don't alter this cell so you can see what your answer should look like.
```


![png](output_44_0.png)



![png](output_44_1.png)


As we can see the transformation of `np.log1p(y)` makes our SalePrice data much more normal and minimizes skew. This is better for linear regression.

So what does this mean? It means that every price will be transformed such that $y \rightarrow \ln(y+1)$

This also means that when our model predicts values, we must transform it back to its original form to get the predicted price, using the inverse function $\ln(y+1) \rightarrow e^{y}-1$, or 
`np.exp(y)-1`.

# MODEL BUILDING

Ridge regression is one of the most common regularization techniques used in regession tasks. It is easy to use, and we'll let it be our model we work with. We need to fit it to our data to tune it's hyperparameter. We can find the right hyperparameter using cross validation.

* In the following cell, import Ridge and cross_val_score
* Create a function that returns the square root of a five fold cross validation of the negative value for the scoring metric. `scoring="neg_mean_squared_error"`
* Using the `np.linspace()` function, create a series of values, between 50 and 70, that we will test as our hyperparameter alpha.
* Create another series that finds that average cross validation score for each alpha.
* Make a matplotlib plot that has the alphas on the x-axis, and the RMSE on the vertical.
* For this to work, you should have used `StandardScaler()` to transform the features, and `np.log1p(y)` on the `SalePrice`.
* Find the minimum value of alpha - this will be our hyperparameter.

If you did everything right, it should the plots should look like the next one:


```python
#Code your solution here.
```


```python
#Don't alter this cell so you can see what your answer should look like.
```

    Best alpha for ridge is:  62.6530612244898



![png](output_48_1.png)


# MODEL DEPLOYMENT AND TESTING
So now that we've cleaned data, selected the features, transformed them and found the right hyperparameter for our model, it's time to use test our model.
Heres what we do next:

* Do a `train_test_split` with a 77-33% split to test
* Convert the predicted values into actual prices via the inverse function, `np.exp(y_pred)+1`, and convert the test values of y to actual prices.
* Plot the predicted prices vs the actual prices 
* Plot the residuals of the predictions using the `ResidualsPlot` from the `yellowbricks.regressor` package (install it if you dont have it)!
* Using the yellow bricks plot, observe the normality of errors and R2 score of the train and test set!


```python
#Code your solution here.
```


```python
#Don't alter this cell so you can see what your answer should look like.
```

    If you did it correctly, your plot should looks like this:



![png](output_51_1.png)



```python
#Code your solution here.
```


```python
#Don't alter this cell so you can see what your answer should look like.
```

    Your plot should look like this:



![png](output_53_1.png)


Lastly, make a horizontal bar plot of the coefficients in the ridge model.
* Use the attribute `model.coef_` to get the coefficients. Put them into a series


```python
#Code your solution here.
```


```python
#Don't alter this cell so you can see what your answer should look like.
```

    Your plot should look like this.



![png](output_56_1.png)


# Sources & Credits
Some of these tutorials were very helpful in writing this lesson.

* https://www.kaggle.com/c/house-prices-advanced-regression-techniques#tutorials
* https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
* https://www.kaggle.com/dgawlik/house-prices-eda
* https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
* https://www.kaggle.com/apapiu/regularized-linear-models
