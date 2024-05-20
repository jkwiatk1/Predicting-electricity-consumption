import pandas as pd


def prepare_weather_data(path):
    # Load data from csv skipping first rows
    warszawa_raw = pd.read_csv(path + 'Warszawa.csv', skiprows=10, header=None)
    pila = pd.read_csv(path + 'Pila.csv', skiprows=10, header=None)
    czestochowa = pd.read_csv(path + 'Czestochowa.csv', skiprows=10, header=None)

    # Only data without time
    warszawa = warszawa_raw.iloc[:, 1:]
    pila = pila.iloc[:, 1:]
    czestochowa = czestochowa.iloc[:, 1:]

    # Check size of dataframes
    assert warszawa.shape == pila.shape == czestochowa.shape, "Wszystkie pliki CSV muszą mieć te same wymiary"

    # Count mean of every field
    average_values = (warszawa + pila + czestochowa) / 3

    # Add time column
    average_values.insert(0, 'Time', warszawa_raw.iloc[:, 0])
    # average_values.insert(0, 'Time', range(1, len(average_values) + 1))

    # Load full data from csv
    data = pd.read_csv(path + "Warszawa.csv", delimiter=';', header=None)

    # Get names of parameters
    parameters_part1 = data.iloc[4].values[0].split(',')
    parameters_part2 = data.iloc[6].values[0].split(',')
    units = data.iloc[5].values[0].split(',')
    parameters = [f"{p1}_{p2}[{p3}]" for p1, p2, p3 in zip(parameters_part1, parameters_part2, units)]
    parameters[0] = average_values.columns[0]
    average_values.columns = parameters
    print(parameters)

    # Write data Polska.csv
    average_values.to_csv(path + 'Polska.csv', index=False, header=True)


def prepare_energy_data(path):
    # Load data from csv skipping first rows
    data2021 = pd.read_csv(path + 'monthly_hourly_load_values_2021.csv', sep=';')
    data2022 = pd.read_csv(path + 'monthly_hourly_load_values_2022.csv', sep=';')
    data2023 = pd.read_csv(path + 'monthly_hourly_load_values_2023.csv', sep=',')

    # Only Poland
    filtered2021 = data2021[data2021['CountryCode'] == 'PL']
    filtered2022 = data2022[data2022['CountryCode'] == 'PL']
    filtered2023 = data2023[data2023['CountryCode'] == 'PL']

    # Only two collumns
    data_selected2021 = filtered2021[['Value']]
    data_selected2022 = filtered2022[['Value']]
    data_selected2023 = filtered2023[['Value']]
    data_selected2021.rename(columns={'Value': 'Energy'}, inplace=True)
    data_selected2022.rename(columns={'Value': 'Energy'}, inplace=True)
    data_selected2023.rename(columns={'Value': 'Energy'}, inplace=True)


    combined_data = pd.concat([data_selected2021, data_selected2022, data_selected2023], ignore_index=True)
    #combined_data.insert(0, 'Time', range(1, len(combined_data) + 1))

    # Write data Polska.csv
    combined_data.to_csv(path + 'Energy.csv', index=False, header=True)


def load_dataset(path=""):
    prepare_weather_data(path)
    prepare_energy_data(path)
    weather = pd.read_csv(path + 'Polska.csv')
    energy = pd.read_csv(path + 'Energy.csv')
    combined_data = pd.concat([weather, energy], axis=1)
    return combined_data


if __name__ == "__main__":
    load_dataset()