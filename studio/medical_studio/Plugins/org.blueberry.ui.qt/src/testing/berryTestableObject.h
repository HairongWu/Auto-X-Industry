/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/


#ifndef BERRYTESTABLEOBJECT_H_
#define BERRYTESTABLEOBJECT_H_

#include <berryObject.h>
#include <berryMacros.h>

#include "berryITestHarness.h"
#include <Poco/Runnable.h>

namespace berry {

/**
 * A testable object.
 * Allows a test harness to register itself with a testable object.
 * The test harness is notified of test-related lifecycle events,
 * such as when is an appropriate time to run tests on the object.
 * This also provides API for running tests as a runnable, and for signaling
 * when the tests are starting and when they are finished.
 * <p>
 * The workbench provides an implementation of this facade, available
 * via <code>PlatformUI.getTestableObject()</code>.
 * </p>
 *
 * @since 3.0
 */
class BERRY_UI_QT TestableObject : public virtual Object {

private:

  ITestHarness::Pointer testHarness;

public:

  berryObjectMacro(TestableObject);

    /**
     * Returns the test harness, or <code>null</code> if it has not yet been set.
     *
     * @return the test harness or <code>null</code>
     */
   virtual ITestHarness::Pointer GetTestHarness() const;

    /**
     * Sets the test harness.
     *
     * @param testHarness the test harness
     */
    virtual void SetTestHarness(ITestHarness::Pointer testHarness);

    /**
     * Runs the given test runnable.
     * The default implementation simply invokes <code>run</code> on the
     * given test runnable.  Subclasses may extend.
     *
     * @param testRunnable the test runnable to run
     */
    virtual void RunTest(Poco::Runnable* testRunnable);

    /**
     * Notification from the test harness that it is starting to run
     * the tests.
     * The default implementation does nothing.
     * Subclasses may override.
     */
    virtual void TestingStarting();

    /**
     * Notification from the test harness that it has finished running the
     * tests.
     * The default implementation does nothing.
     * Subclasses may override.
     */
    virtual void TestingFinished();
};

}

#endif /* BERRYTESTABLEOBJECT_H_ */
