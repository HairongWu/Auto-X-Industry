/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkCoreServices_h
#define mitkCoreServices_h

#include "MitkCoreExports.h"

#include <mitkCommon.h>
#include <mitkLog.h>

#include <mitkServiceInterface.h>
#include <usGetModuleContext.h>
#include <usModuleContext.h>
#include <usServiceReference.h>

#include <cassert>

namespace mitk
{
  struct IMimeTypeProvider;
  class IPropertyAliases;
  class IPropertyDescriptions;
  class IPropertyDeserialization;
  class IPropertyExtensions;
  class IPropertyFilters;
  class IPropertyPersistence;
  class IPropertyRelations;
  class IPreferencesService;

  /**
   * @brief Access MITK core services.
   *
   * This class can be used to conveniently access common
   * MITK Core service objects. Some getter methods where implementations
   * exist in the core library are guaranteed to return a non-nullptr service object.
   *
   * To ensure that CoreServices::Unget() is called after the caller
   * has finished using a service object, you should use the CoreServicePointer
   * helper class which calls Unget() when it goes out of scope:
   *
   * \code
   * CoreServicePointer<IShaderRepository> shaderRepo(CoreServices::GetShaderRepository());
   * // Do something with shaderRepo
   * \endcode
   *
   * @see CoreServicePointer
   */
  class MITKCORE_EXPORT CoreServices
  {
  public:

    /**
     * @brief Get an IPropertyAliases instance.
     * @param context The module context of the module getting the service.
     * @return A non-nullptr IPropertyAliases instance.
     */
    static IPropertyAliases *GetPropertyAliases(us::ModuleContext *context = us::GetModuleContext());

    /**
     * @brief Get an IPropertyDescriptions instance.
     * @param context The module context of the module getting the service.
     * @return A non-nullptr IPropertyDescriptions instance.
     */
    static IPropertyDescriptions *GetPropertyDescriptions(us::ModuleContext *context = us::GetModuleContext());

    /**
     * @brief Get an IPropertyDeserialization instance.
     * @param context The module context of the module getting the service.
     * @return A non-nullptr IPropertyDeserialization instance.
     */
    static IPropertyDeserialization* GetPropertyDeserialization(us::ModuleContext* context = us::GetModuleContext());

    /**
     * @brief Get an IPropertyExtensions instance.
     * @param context The module context of the module getting the service.
     * @return A non-nullptr IPropertyExtensions instance.
     */
    static IPropertyExtensions *GetPropertyExtensions(us::ModuleContext *context = us::GetModuleContext());

    /**
     * @brief Get an IPropertyFilters instance.
     * @param context The module context of the module getting the service.
     * @return A non-nullptr IPropertyFilters instance.
     */
    static IPropertyFilters *GetPropertyFilters(us::ModuleContext *context = us::GetModuleContext());

    /**
    * @brief Get an IPropertyPersistence instance.
    * @param context The module context of the module getting the service.
    * @return A non-nullptr IPropertyPersistence instance.
    */
    static IPropertyPersistence *GetPropertyPersistence(us::ModuleContext *context = us::GetModuleContext());

    /**
    * @brief Get an IPropertyRelations instance.
    * @param context The module context of the module getting the service.
    * @return A non-nullptr IPropertyRelations instance.
    */
    static IPropertyRelations *GetPropertyRelations(us::ModuleContext *context = us::GetModuleContext());

    /**
     * @brief Get an IMimeTypeProvider instance.
     * @param context The module context of the module getting the service.
     * @return A non-nullptr IMimeTypeProvider instance.
     */
    static IMimeTypeProvider *GetMimeTypeProvider(us::ModuleContext *context = us::GetModuleContext());

    /**
     * @brief Get an IPreferencesService instance.
     * @param context The module context of the module getting the service.
     * @return A non-nullptr IPreferencesService instance.
     * @sa IPreferences
     */
    static IPreferencesService *GetPreferencesService(us::ModuleContext *context = us::GetModuleContext());

    /**
     * @brief Unget a previously acquired service instance.
     * @param service The service instance to be released.
     * @param context
     * @return \c true if ungetting the service was successful, \c false otherwise.
     */
    template <class S>
    static bool Unget(S *service, us::ModuleContext *context = us::GetModuleContext())
    {
      return Unget(context, us_service_interface_iid<S>(), service);
    }

  private:
    static bool Unget(us::ModuleContext *context, const std::string &interfaceId, void *service);

    // purposely not implemented
    CoreServices();
    CoreServices(const CoreServices &);
    CoreServices &operator=(const CoreServices &);
  };

  /**
   * @brief A RAII helper class for core service objects.
   *
   * This is class is intended for usage in local scopes; it calls
   * CoreServices::Unget(S*) in its destructor. You should not construct
   * multiple CoreServicePointer instances using the same service pointer,
   * unless it is retrieved by a new call to a CoreServices getter method.
   *
   * @see CoreServices
   */
  template <class S>
  class MITK_LOCAL CoreServicePointer
  {
  public:
    explicit CoreServicePointer(S *service, us::ModuleContext* context = us::GetModuleContext())
      : m_Service(service),
        m_Context(context)
    {
      assert(service);
    }

    ~CoreServicePointer()
    {
      try
      {
        CoreServices::Unget(m_Service, m_Context);
      }
      catch (const std::exception &e)
      {
        MITK_ERROR << e.what();
      }
      catch (...)
      {
        MITK_ERROR << "Ungetting core service failed.";
      }
    }

    S *operator->() const
    {
      return m_Service;
    }

  private:
    S *const m_Service;
    us::ModuleContext* m_Context;
  };
}

#endif
