import pandas as pd
import numpy as np
from enum import Enum


class Gender(Enum):
    MALE = 0
    FEMALE = 1
    OTHER = 2


class Married(Enum):
    NO = 0
    YES = 1


class WorkType(Enum):
    PRIVATE = 0
    SELF_EMP = 1
    GOVT_JOB = 2
    CHILDREN = 3
    NEVER_WORKED = 4


class Residence(Enum):
    RURAL = 0
    URBAN = 1


class SmokingStatus(Enum):
    NEVER = 0
    UNKNOWN = 1
    FORMERLY_SMOKED = 2
    SMOKES = 3


def load(filepath):
    file = pd.read_csv(filepath)
    file['gender'].mask(file['gender'] == 'Female', Gender.FEMALE.value, inplace=True)
    file['gender'].mask(file['gender'] == 'Male', Gender.MALE.value, inplace=True)
    file['gender'].mask(file['gender'] == 'Other', Gender.OTHER.value, inplace=True)
    file['ever_married'].mask(file['ever_married'] == 'Yes', Married.YES.value, inplace=True)
    file['ever_married'].mask(file['ever_married'] == 'No', Married.NO.value, inplace=True)
    file['work_type'].mask(file['work_type'] == 'Private', WorkType.PRIVATE.value, inplace=True)
    file['work_type'].mask(file['work_type'] == 'Self-employed', WorkType.SELF_EMP.value, inplace=True)
    file['work_type'].mask(file['work_type'] == 'Govt_job', WorkType.GOVT_JOB.value, inplace=True)
    file['work_type'].mask(file['work_type'] == 'children', WorkType.CHILDREN.value, inplace=True)
    file['work_type'].mask(file['work_type'] == 'Never_worked', WorkType.NEVER_WORKED.value, inplace=True)
    file['Residence_type'].mask(file['Residence_type'] == 'Urban', Residence.URBAN.value, inplace=True)
    file['Residence_type'].mask(file['Residence_type'] == 'Rural', Residence.RURAL.value, inplace=True)
    file['smoking_status'].mask(file['smoking_status'] == 'smokes', SmokingStatus.SMOKES.value, inplace=True)
    file['smoking_status'].mask(file['smoking_status'] == 'never smoked', SmokingStatus.NEVER.value, inplace=True)
    file['smoking_status'].mask(file['smoking_status'] == 'formerly smoked', SmokingStatus.FORMERLY_SMOKED.value,
                                inplace=True)
    file['smoking_status'].mask(file['smoking_status'] == 'Unknown', SmokingStatus.UNKNOWN.value, inplace=True)
    file['bmi'].mask(np.isnan(file['bmi']), 0, inplace=True)

    file.drop(labels='id', axis=1, inplace=True)

    for idx, column in file.iteritems():
        if idx == 'stroke':
            break
        mean = column.mean()
        std = column.std()
        file[idx] = (file[idx] - mean) / std

    file.to_csv("normalized_data.csv")


load('../healthcare-dataset-stroke-data.csv')
