package edu.stanford.bmir.protege.web.server.dispatch;


import edu.stanford.bmir.protege.web.server.access.AccessManager;
import edu.stanford.bmir.protege.web.server.dispatch.validators.CompositeRequestValidator;
import edu.stanford.bmir.protege.web.server.dispatch.validators.NullValidator;
import edu.stanford.bmir.protege.web.server.dispatch.validators.ProjectPermissionValidator;
import edu.stanford.bmir.protege.web.shared.access.ActionId;
import edu.stanford.bmir.protege.web.shared.access.BuiltInAction;
import edu.stanford.bmir.protege.web.shared.dispatch.Action;
import edu.stanford.bmir.protege.web.shared.dispatch.ProjectAction;
import edu.stanford.bmir.protege.web.shared.dispatch.Result;
import edu.stanford.bmir.protege.web.shared.project.HasProjectId;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static com.google.common.base.Preconditions.checkNotNull;

/**
 * Author: Matthew Horridge<br>
 * Stanford University<br>
 * Bio-Medical Informatics Research Group<br>
 * Date: 21/02/2013
 * <p>
 * A skeleton handler for handling actions that pertain to projects (i.e. {@link Action}s that implement
 * {@link HasProjectId}. Further more, the validation includes a check to see if the project
 * actually exists and fails if this isn't the case.
 * </p>
 */
public abstract class AbstractProjectActionHandler<A extends ProjectAction<R>, R extends Result> implements ProjectActionHandler<A, R> {

    @Nonnull
    private final AccessManager accessManager;

    public AbstractProjectActionHandler(@Nonnull AccessManager accessManager) {
        this.accessManager = checkNotNull(accessManager);
    }

    @Nonnull
    @Override
    public final RequestValidator getRequestValidator(@Nonnull A action, @Nonnull RequestContext requestContext) {
        List<RequestValidator> validators = new ArrayList<>();

        BuiltInAction builtInAction = getRequiredExecutableBuiltInAction(action);
        if(builtInAction != null) {
            ProjectPermissionValidator validator = new ProjectPermissionValidator(accessManager,
                                                                                  action.getProjectId(),
                                                                                  requestContext.getUserId(),
                                                                                  builtInAction.getActionId());
            validators.add(validator);
        }



        ActionId reqActionId = getRequiredExecutableAction();
        if (reqActionId != null) {
            ProjectPermissionValidator validator = new ProjectPermissionValidator(accessManager,
                                                                                  action.getProjectId(),
                                                                                  requestContext.getUserId(),
                                                                                  reqActionId);
            validators.add(validator);
        }

        Iterable<BuiltInAction> requiredExecutableBuiltInActions = getRequiredExecutableBuiltInActions(action);
        for(BuiltInAction actionId : requiredExecutableBuiltInActions) {
            ProjectPermissionValidator validator = new ProjectPermissionValidator(accessManager,
                                                                                  action.getProjectId(),
                                                                                  requestContext.getUserId(),
                                                                                  actionId.getActionId());
            validators.add(validator);
        }

        final RequestValidator additionalRequestValidator = getAdditionalRequestValidator(action, requestContext);
        if (additionalRequestValidator != NullValidator.get()) {
            validators.add(additionalRequestValidator);
        }
        return CompositeRequestValidator.get(validators);
    }

    @Nullable
    protected BuiltInAction getRequiredExecutableBuiltInAction(A action) {
        return null;
    }



    @Nullable
    protected ActionId getRequiredExecutableAction() {
        return null;
    }

    @Nonnull
    protected Iterable<BuiltInAction> getRequiredExecutableBuiltInActions(A action) {
        return Collections.emptyList();
    }

    /**
     * Gets an additional validator that is specific to the implementing handler.  This is returned as part of a
     * {@link CompositeRequestValidator} by the the implementation of
     * the {@link #getRequestValidator(edu.stanford.bmir.protege.web.shared.dispatch.Action,
     * edu.stanford.bmir.protege.web.server.dispatch.RequestContext)} method.
     *
     * @param action         The action that the validation will be completed against.
     * @param requestContext The {@link RequestContext} that describes the context for the request.
     * @return A {@link RequestValidator} for this handler.  Not {@code null}.
     */
    @Nonnull
    protected RequestValidator getAdditionalRequestValidator(A action, RequestContext requestContext) {
        return NullValidator.get();
    }
}
