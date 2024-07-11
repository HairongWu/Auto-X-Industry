/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkStandardFileLocations.h"
#include "mitkTestingMacros.h"
#include "mitkPointSetStatisticsCalculator.h"

//#include <QtCore>

/**
 * \brief Test class for mitkPointSetStatisticsCalculator
 */
class mitkPointSetStatisticsCalculatorTestClass
{
public:

  static void TestInstantiation()
    {
    // let's create an object of our class
    mitk::PointSetStatisticsCalculator::Pointer myPointSetStatisticsCalculator = mitk::PointSetStatisticsCalculator::New();
    MITK_TEST_CONDITION_REQUIRED(myPointSetStatisticsCalculator.IsNotNull(),"Testing instantiation with constructor 1.");

    mitk::PointSet::Pointer myTestPointSet = mitk::PointSet::New();
    mitk::PointSetStatisticsCalculator::Pointer myPointSetStatisticsCalculator2 = mitk::PointSetStatisticsCalculator::New(myTestPointSet);
    MITK_TEST_CONDITION_REQUIRED(myPointSetStatisticsCalculator2.IsNotNull(),"Testing instantiation with constructor 2.");
    }

static void TestSimpleCase()
    {

    MITK_TEST_OUTPUT(<< "Starting simple test case...");

    mitk::Point3D test;
    mitk::PointSet::Pointer testPointSet = mitk::PointSet::New();

    mitk::FillVector3D(test,0,0,0);
    testPointSet->InsertPoint(0,test);

    mitk::FillVector3D(test,1,1,1);
    testPointSet->InsertPoint(1,test);

    mitk::PointSetStatisticsCalculator::Pointer myPointSetStatisticsCalculator = mitk::PointSetStatisticsCalculator::New(testPointSet);

    MITK_TEST_CONDITION_REQUIRED((myPointSetStatisticsCalculator->GetPositionMean()[0]==0.5),".. Testing GetPositionMean");
    MITK_TEST_CONDITION_REQUIRED((myPointSetStatisticsCalculator->GetPositionStandardDeviation()[0]==0.5),".. Testing GetPositionStandardDeviation");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionSampleStandardDeviation()[0],0.70710678118654757),".. Testing GetPositionSampleStandardDeviation");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionErrorMean(),0.8660254, 1E-5),".. Testing GetPositionErrorMean");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionErrorRMS(),0.8660254, 1E-5),".. Testing GetPositionErrorRMS");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionErrorMax(),0.8660254, 1E-5),".. Testing GetPositionErrorMax");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionErrorMedian(),0.8660254, 1E-5),".. Testing GetPositionErrorMedian");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionErrorMin(),0.8660254, 1E-5),".. Testing GetPositionErrorMin");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionErrorSampleStandardDeviation(),0, 1E-5),".. Testing GetPositionErrorSampleStandardDeviation");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionErrorStandardDeviation(),0, 1E-5),".. Testing GetPositionErrorStandardDeviation");
    }

static void TestComplexCase()
    {

    MITK_TEST_OUTPUT(<< "Starting complex test case...");
    mitk::Point3D testPoint;
    mitk::PointSet::Pointer testPointSet = mitk::PointSet::New();

    //1st point
    mitk::FillVector3D(testPoint,0,1,0);
    testPointSet->InsertPoint(0,testPoint);

    //2nd point
    mitk::FillVector3D(testPoint,0,1,0.34);
    testPointSet->InsertPoint(1,testPoint);

    //3rd point
    mitk::FillVector3D(testPoint,1,0.5,1);
    testPointSet->InsertPoint(2,testPoint);

    //4th point
    mitk::FillVector3D(testPoint,15,3,2);
    testPointSet->InsertPoint(3,testPoint);

    //5th point
    mitk::FillVector3D(testPoint,2,22.5,1.2655);
    testPointSet->InsertPoint(4,testPoint);

    //6th point
    mitk::FillVector3D(testPoint,4,1.3,2);
    testPointSet->InsertPoint(5,testPoint);

    //7th point
    mitk::FillVector3D(testPoint,0.001,0,1);
    testPointSet->InsertPoint(6,testPoint);

    //8th point
    mitk::FillVector3D(testPoint,1.2525,2.22,3);
    testPointSet->InsertPoint(7,testPoint);

    //9th point
    mitk::FillVector3D(testPoint,3.1,3,1);
    testPointSet->InsertPoint(8,testPoint);

    mitk::PointSetStatisticsCalculator::Pointer myPointSetStatisticsCalculator = mitk::PointSetStatisticsCalculator::New();
    myPointSetStatisticsCalculator->SetPointSet(testPointSet);

    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionMean()[2],1.2895, 1E-5),".. Testing GetPositionMean");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionStandardDeviation()[2],0.86614074, 1E-5),".. Testing GetPositionStandardDeviation");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionSampleStandardDeviation()[2],0.91868098, 1E-5),".. Testing GetPositionStandardDeviation");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionErrorMean(),6.06656587, 1E-5),".. Testing GetPositionErrorMean");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionErrorRMS(),8.0793161, 1E-5),".. Testing GetPositionErrorRMS");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionErrorMax(),18.6875241, 1E-5),".. Testing GetPositionErrorMax");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionErrorMedian(),4.18522229, 1E-5),".. Testing GetPositionErrorMedian");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionErrorMin(),0.90082741, 1E-5),".. Testing GetPositionErrorMin");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionErrorSampleStandardDeviation(),5.65960626, 1E-5),".. Testing GetPositionErrorSampleStandardDeviation");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(myPointSetStatisticsCalculator->GetPositionErrorStandardDeviation(),5.33592795, 1E-5),".. Testing GetPositionErrorStandardDeviation");

    }


};

int mitkPointSetStatisticsCalculatorTest(int, char* [])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkPointSetStatisticsCalculatorTest")

  mitkPointSetStatisticsCalculatorTestClass::TestInstantiation();
  mitkPointSetStatisticsCalculatorTestClass::TestSimpleCase();
  mitkPointSetStatisticsCalculatorTestClass::TestComplexCase();

  MITK_TEST_END()
}
