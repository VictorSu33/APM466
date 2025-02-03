import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

bond_pd = pd.read_csv('BondDataAPM466(Sheet1).csv', sep=',')

def get_percentage_from_string(percent):
    return float(percent.strip('%'))

bond_pd['Coupon '] = bond_pd['Coupon '].apply(get_percentage_from_string)

def convert_dirty_price(df):
    #convert clean prices to dirty
    for i in range(0,43):
        maturity = datetime.strptime(df.iloc[i,3], "%m/%d/%Y").date()
        coupon = df.iloc[i,1]
        for j in range(4,14):
            current = datetime.strptime(df.columns.values[j], "%m/%d/%Y").date()
            last_payment = get_last_payment(maturity, current)
            delta = (current - last_payment).days
            if delta >= 183:
                raise ValueError
            accrued_interest = coupon * delta/365
            df.iloc[i,j] += accrued_interest
            df.iloc[i,j] = round(df.iloc[i,j],2)

def get_last_payment(maturity, current):
    #find last payment given maturity
    if maturity.month == 1 and maturity.day < current.day:
        return datetime(2025, maturity.month, maturity.day).date()
    elif datetime(2024, maturity.month, maturity.day).date() > current + relativedelta(months=-6):
        return datetime(2024, maturity.month, maturity.day).date()
    else:
        return datetime(2024, maturity.month, maturity.day).date() + relativedelta(months=+6)

convert_dirty_price(bond_pd)

ten_bonds_for_curve = [25, 32, 40,41,19,22,12,1,9,3] 

# chosen bonds starting at 6/1 
# []

# look for other possible groups
# print(bond_pd.groupby(["Maturity Month"])["Maturity Date"].apply(list))

bond_pd["Maturity Month"] = bond_pd["Maturity Date"].apply(lambda x: int(x[:x.index('/')]))

bootstrap_data = bond_pd.loc[ten_bonds_for_curve].reset_index(drop=True)

def ytm_curve(df):
    data  = []
    all_ytms = []
    all_times = []
    for day in range(4,14): 

        times = []
        ytms = []

        for i in range(0,10): 
            date = datetime.strptime(df.columns.values[day], "%m/%d/%Y").date()
            maturity = datetime.strptime(df.iloc[i,3], "%m/%d/%Y").date()
            times.append((maturity - date).days/365)
            semi_coupon = df.iloc[i,1] * 0.5
            coefficients = [-df.iloc[i,day]]  + [semi_coupon] *  i + [100 + semi_coupon]

            # function for ytm: 0 = -P + sum_i p_i * (1 + r)^-t_i
            func = lambda r: coefficients[0] + sum([coefficients[j] * (1 + r)**(-times[j-1]) for j in range(1,i+2)])

            ytms.append(fsolve(func,0.03))

        interp_yields = np.interp([1,2,3,4,5], np.squeeze(times),np.squeeze(ytms))

        data.append(interp_yields)

        all_times.append(times)
        all_ytms.append(ytms)
    
    for i in range(0,10):
        plt.plot(all_times[i], all_ytms[i], label='YTM curve on ' + df.columns.values[i + 4])

    plt.xlabel("Time to maturity")
    plt.ylabel("YTMs")
    plt.title("YTM curves from Jan 6 to Jan 17")
    plt.legend()
    plt.grid(True)
    plt.show()


        #if day ==4:
        #    print(interp_yields)
        
    return data
        
yields = ytm_curve(bootstrap_data)

def bootstrapping(df):
    #calculate and create spot rate and curve
    data = []

    all_yields = []
    all_times = []
    for day in range(4,14):
        yields = {}
    
        for i in range(0,10):
            date = datetime.strptime(df.columns.values[day], "%m/%d/%Y").date()
            maturity = datetime.strptime(df.iloc[i,3], "%m/%d/%Y").date()
            time_delta =  (maturity - date).days/365
            semi_coupon = df.iloc[i,1] * 0.5
            price = df.iloc[i,day]
            yields[time_delta] = round(calculate_yield(yields,semi_coupon, price, time_delta, i),5)

        x = list(yields.keys())
        y = list(yields.values())

        all_yields.append(y)
        all_times.append(x)
        #if day == 4:
        #    print(yields)

        #get interpolated values
        interp_yields = np.interp([1,2,3,4,5], x,y)

        #if day == 4:
        #    print(interp_yields)

        #calculuate forward yield using r(T_1,T_2) = -log P(t,T_2) + log P(t,T_1) / T_2 - T_1

        forward_rates = []
        one_year = interp_yields[0]

        for i in range(1,5):
            forward_rates.append((interp_yields[i] * (i +1)-one_year)/i)

        #if day == 4:
        #    print(forward_rates)


        data.append(forward_rates)

    labels = ['1Y-1Y', '1Y-2Y', '1Y-3Y', '1Y-4Y']

    for i in range(0,10):
        plt.plot(all_times[i], all_yields[i], label='Spot curve on ' + df.columns.values[i + 4])

    plt.xlabel("Time to maturity")
    plt.ylabel("Spot Rate")
    plt.title("Spot curves from Jan 6 to Jan 17")
    plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(0,10):
        plt.plot(labels, data[i], label='Forward curve on ' + df.columns.values[i + 4])

    plt.xlabel("Forward")
    plt.ylabel("Forward Rate")
    plt.title("Forward curves from Jan 6 to Jan 17")
    plt.legend()
    plt.grid(True)
    plt.show()
    return data

def calculate_yield(yields, semi_coupon,price, time_delta, i):
    cashflows = [semi_coupon] * i  + [100 + semi_coupon]
    
    count = 0
    curr_total = 0
    for time, r in yields.items():
        curr_total += cashflows[count]*np.exp(-r*time)
        count += 1

    return -np.log((price - curr_total)/cashflows[i])/time_delta

forward_rates = bootstrapping(bootstrap_data)

def pca(data):
    #since we are calculating eigenvalues and eigenvectors of covaraince matrix, use unscaled data for PCA
    #pca without dimension reduction calculates covariance of data

    log_returns = log_return(data)
    pca = PCA()
    pca.fit(np.transpose(log_returns))
    print("eigenvectors", pca.components_) #read documentation
    print("eigenvalues" ,  pca.singular_values_**2)
    print("explained varaince", pca.explained_variance_ratio_)
    print("covariance", pca.get_covariance())

    

def log_return(data):
    log_returns = []
    for row in np.transpose(data):
        maturity_returns = []
        for i in range(len(row)-1):
            maturity_returns.append(np.log(row[i + 1]/row[i]))
        log_returns.append(maturity_returns)
    return log_returns

pca(yields)

pca(forward_rates)
