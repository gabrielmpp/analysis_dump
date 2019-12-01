import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from climIndices.tools import get_data
import pandas as pd

##### Reading data #####
lcstimelen = 16
soi = get_data('oni').to_xarray().to_array().drop('variable').squeeze('variable')
ftle_ts = xr.open_dataarray('data/ftle_ts_tiete_16.nc')*4/lcstimelen

precip_ts = xr.open_dataarray('data/pr_ts_tiete_16.nc')
precip_ts = precip_ts
x_dep = xr.open_dataarray('data/x_departure_tiete.nc').values.flatten()
x_dep = x_dep.where(x_dep<180, drop=True)

y_dep = xr.open_dataarray('data/y_departure_tiete.nc').values.flatten()
y_dep = y_dep.where(y_dep<90, drop=True )
soi = soi.interp(time=precip_ts.time)
sflow = pd.read_csv('data/df_vazao_natural_afluente.csv')
sflow = sflow[['time', '243']]
sflow = sflow.set_index('time')
sflow = xr.DataArray(sflow.values[:, 0], dims=['time'], coords={'time': [pd.Timestamp(x) for x in sflow.index.values]})
sflow = sflow.interp(time=ftle_ts.time)

sflow = sflow.rolling(time=lcstimelen).sum()
threshold = 0.8

available_vars = {
    'sflow': sflow,
    'precip': precip_ts
}
##### Start analysis #####
varname = 'precip'
units = 'mm'
var = available_vars[varname]  # or pr_ts
var_cz = var.where(ftle_ts > threshold, drop=True)
var_nocz = var.where(ftle_ts < threshold, drop=True)

frac_cz = var_cz.sum()/var.sum()
frac_nocz = var_cz.sum()/var.sum()

print(f"Average {varname} during strong convergence is {var_cz.mean().round(2).values} and  "
      f"no convergence is {var_nocz.mean().values}")
print(f'Fraction of {varname} associated to CZs is {frac_cz.values}')


ftle_nino = ftle_ts.where(soi > 0.5, drop=True)
ftle_nina = ftle_ts.where(soi < -0.5, drop=True)

var_nino = var.where(soi > 0.5, drop=True)
var_nina = var.where(soi < -0.5, drop=True)
var_nino.plot.hist(log='y')
var_nina.plot.hist(log='y')
plt.legend(['Nino', 'Nine'])
plt.show()
var_nino_cz = var_nino.where(ftle_nino > 1)
var_nino_nocz = var_nino.where(ftle_nino < 1)
var_nina_cz = var_nina.where(ftle_nina > 1)
var_nina_nocz = var_nina.where(ftle_nina < 1)

plt.hist(ftle_nino, bins=20, alpha=0.5, log='y')
plt.hist(ftle_nina, bins=20, alpha=0.5, log='y')
plt.legend(['Nino', 'Nina'])
plt.xlabel('FTLE')
plt.ylabel('# of events')
plt.show()
### FTLE is the same in nino or nina
plt.style.use("ggplot")
log = 'y'
alpha = 0.9
nbins = 30
plt.figure(figsize=[9, 6])
bins = np.linspace(var.quantile(0.0001), var.quantile(0.999), nbins)
plt.hist(var_nocz, bins=bins, alpha=alpha, log=log)

plt.hist(var_cz, bins=bins, alpha=alpha, log=log)


plt.hist(var_nino_cz, linewidth=3.5,
         bins=bins, alpha=alpha, histtype='step', log=log)
plt.hist(var_nina_cz, bins=bins, alpha=alpha, log=log, histtype='step', linewidth=3.5)

plt.legend(['CZ and ONI > 0.5', 'CZ and ONI < -0.5', 'No CZ', 'CZ'])
plt.xlabel(f'{varname} ({units})')
#plt.ylim([1e-1, 1e4])
plt.ylabel('# of events')
plt.tight_layout()
plt.savefig(f'histogram_{varname}.pdf')
plt.close()

quants = np.arange(0, 1, 0.01)
q_list_cz = []
q_list_nocz = []
q_list_nino_cz = []
q_list_nina_cz = []
q_list_cz_boostrap = []
q_list_nocz_bootstrap = []
q_list_nina_cz_bootstrap = []
q_list_nino_cz_bootstrap = []


bslen = 50
nvalues = 500
var_cz_arr = np.zeros([bslen, quants.shape[0]])
var_nocz_arr = np.zeros([bslen, quants.shape[0]])
var_nina_cz_arr = np.zeros([bslen, quants.shape[0]])
var_nino_cz_arr = np.zeros([bslen, quants.shape[0]])

for j, quant in enumerate(quants):
    for i, _ in enumerate(np.arange(bslen)):
        var_cz_arr[i, j] = np.quantile(np.random.choice(var_cz.values, nvalues), quant)
        var_nocz_arr[i, j] = np.quantile(np.random.choice(var_nocz.values, nvalues), quant)
        var_nina_cz_arr[i, j] = np.quantile(np.random.choice(var_nina_cz.values, nvalues), quant)
        var_nino_cz_arr[i, j] = np.quantile(np.random.choice(var_nino_cz.values, nvalues), quant)

var_mean_cz_bs = np.mean(var_cz_arr, axis=0)
var_mean_nocz_bs = np.mean(var_nocz_arr, axis=0)
stdev_var_mean_cz = 2 * (np.var(var_cz_arr, axis=0) ** 0.5)
stdev_var_mean_nocz = 2 * (np.var(var_nocz_arr, axis=0) ** 0.5)

cz_bs_array = np.array(q_list_cz_boostrap)
var_cz.quantile(0.5)
var_nocz.quantile(0.5)


plt.style.use('seaborn')
plt.plot([0, 20], [0, 20], color='black', linestyle='dashed')
plt.errorbar(var_mean_nocz_bs, var_mean_cz_bs, xerr=stdev_var_mean_nocz, yerr=stdev_var_mean_nocz, ecolor='gray',
             errorevery=1, elinewidth=0.3, linewidth=0.02)
plt.scatter(var_mean_nocz_bs, var_mean_cz_bs)
plt.legend(['1:1'])
plt.xlim([0, 22])
plt.ylim([0, 22])

median_y = var_cz.quantile(0.5)
median_x = var_nocz.quantile(0.5)
extreme_x = var_nocz.quantile(0.9)
extreme_y = var_cz.quantile(0.9)
plt.text(median_x+0.2, median_y, "median", bbox=dict(boxstyle="round4,pad=.1", fc="0.8"))
#plt.scatter(median_x, median_y, color='black')

plt.text(extreme_x+0.2, extreme_y, "90% quantile", bbox=dict(boxstyle="round4,pad=.1", fc="0.8"))
#plt.scatter(extreme_x, extreme_y, color='black')

plt.ylabel('Quantiles of precip. during CZ events (mm)')
plt.xlabel('Quantiles of precip. during non-CZ events (mm)')
plt.savefig('/run/media/gab/gab_hd/phd_panels/qqplot.pdf')
plt.close()

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression as LR
import cartopy.crs as ccrs
x = x_dep.where(x_dep<-10, drop=True).where(x_dep>-180,drop=True)
y = y_dep.where(y_dep<-1, drop=True).where(y_dep>-90,drop=True)
z = ftle_ts
z = z.sel(time=x.time.values, method='nearest')
scaler = MinMaxScaler()
scaler = scaler.fit(X=np.stack([x.values, y.values, z.values], axis=1))
input_data = scaler.transform(X=np.stack([x, y, z], axis=1))
fig = plt.figure(figsize=[20, 20])

ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

c = ax.scatter(x.values, y.values, c=z.values, s=30*z.values**2, cmap="plasma", transform=ccrs.PlateCarree(), alpha=0.9)
fig.colorbar(c)
ax.coastlines()
plt.show()