/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef BERRYPARTTESTER_H_
#define BERRYPARTTESTER_H_

#include "berryIEditorPart.h"
#include "berryIEditorInput.h"
#include "berryIViewPart.h"

namespace berry {

/**
 * \ingroup org_blueberry_ui_internal
 *
 */
class PartTester {
private:
  PartTester() { }

    /**
     * Sanity-check the public interface of the editor. This is called on every editor after it
     * is fully initialized, but before it is actually connected to the editor reference or the
     * layout. Calls as much of the editor's public interface as possible to test for exceptions,
     * and tests the return values for glaring faults. This does not need to be an exhaustive conformance
     * test, as it is called every time an editor is opened and it needs to be efficient.
     * The part should be unmodified when the method exits.
     *
     * @param part
     */
public: static void TestEditor(IEditorPart::Pointer part);

public: static void TestEditorInput(IEditorInput::Pointer input);

    /**
     * Sanity-checks a workbench part. Exercises the public interface and tests for any
     * obviously bogus return values. The part should be unmodified when the method exits.
     *
     * @param part
     * @throws Exception
     */
private: static void TestWorkbenchPart(IWorkbenchPart::Pointer part);


    /**
     * Sanity-check the public interface of a view. This is called on every view after it
     * is fully initialized, but before it is actually connected to the part reference or the
     * layout. Calls as much of the part's public interface as possible without modifying the part
     * to test for exceptions and check the return values for glaring faults. This does not need
     * to be an exhaustive conformance test, as it is called every time an editor is opened and
     * it needs to be efficient.
     *
     * @param part
     */
public: static void TestView(IViewPart::Pointer part);
};

}

#endif /*BERRYPARTTESTER_H_*/
