/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef BERRYIADAPTERFACTORY_H_
#define BERRYIADAPTERFACTORY_H_

#include <org_blueberry_core_runtime_Export.h>

#include <vector>
#include <typeinfo>

namespace berry {

/**
 * An adapter factory defines behavioral extensions for
 * one or more classes that implements the <code>IAdaptable</code>
 * interface. Adapter factories are registered with an
 * adapter manager.
 * <p>
 * This interface can be used without OSGi running.
 * </p><p>
 * Clients may implement this interface.
 * </p>
 * @see IAdapterManager
 * @see IAdaptable
 */
struct org_blueberry_core_runtime_EXPORT IAdapterFactory {

  virtual ~IAdapterFactory() {};

  /**
   * Returns an object which can be cast to the given adapter type and which is
   * associated with the given adaptable object. Returns <code>0</code> if
   * no such object can be found.
   *
   * A typical implementation would look like this:
   *
   * <code>
   * void* GetAdapter(void* adaptableObject, const std::type_info& adaptableType, const std::string& adapterType)
   * {
   *   if (Image* img = CastHelper<Image>(adaptableObject, adaptableType))
   *   {
   *     if (adapterType == "berry::IResource")
   *     {
   *       return new IResource(img->GetPath());
   *     }
   *   }
   *   return 0;
   * }
   * </code>
   *
   * @param adaptableObject the adaptable object being queried
   *   (usually an instance of <code>IAdaptable</code>)
   * @param adapterType the type of adapter to look up
   * @return a object castable to the given adapter type,
   *    or <code>0</code> if this adapter factory
   *    does not have an adapter of the given type for the
   *    given object
   */
  virtual Object* GetAdapter(IAdaptable* adaptableObject, const std::string& adapterType) = 0;

  /**
   * Returns the collection of adapter types handled by this
   * factory.
   * <p>
   * This method is generally used by an adapter manager
   * to discover which adapter types are supported, in advance
   * of dispatching any actual <code>getAdapter</code> requests.
   * </p>
   *
   * @param[out] adapters the collection of adapter types
   */
  virtual void GetAdapterList(std::vector<const std::type_info&>& adapters) = 0;
};

}
#endif /*BERRYIADAPTERFACTORY_H_*/
