package edu.stanford.bmir.protege.web.server.user;

import edu.stanford.bmir.protege.web.server.access.AccessManager;
import edu.stanford.bmir.protege.web.server.auth.AuthenticationManager;
import edu.stanford.bmir.protege.web.server.dispatch.ApplicationActionHandler;
import edu.stanford.bmir.protege.web.server.dispatch.ExecutionContext;
import edu.stanford.bmir.protege.web.server.dispatch.RequestContext;
import edu.stanford.bmir.protege.web.server.dispatch.RequestValidator;
import edu.stanford.bmir.protege.web.server.dispatch.validators.ApplicationPermissionValidator;
import edu.stanford.bmir.protege.web.shared.user.CreateUserAccountAction;
import edu.stanford.bmir.protege.web.shared.user.CreateUserAccountResult;

import javax.annotation.Nonnull;
import javax.inject.Inject;

import static com.google.common.base.Preconditions.checkNotNull;
import static edu.stanford.bmir.protege.web.shared.access.BuiltInAction.CREATE_ACCOUNT;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 19/02/15
 */
public class CreateUserAccountActionHandler implements ApplicationActionHandler<CreateUserAccountAction, CreateUserAccountResult> {

    @Nonnull
    private final AccessManager accessManager;

    @Nonnull
    private final AuthenticationManager authenticationManager;

    @Inject
    public CreateUserAccountActionHandler(@Nonnull AccessManager accessManager,
                                          @Nonnull AuthenticationManager authenticationManager) {
        this.accessManager = checkNotNull(accessManager);
        this.authenticationManager = checkNotNull(authenticationManager);
    }

    @Nonnull
    @Override
    public Class<CreateUserAccountAction> getActionClass() {
        return CreateUserAccountAction.class;
    }

    @Nonnull
    @Override
    public RequestValidator getRequestValidator(@Nonnull CreateUserAccountAction action, @Nonnull RequestContext requestContext) {
        return new ApplicationPermissionValidator(accessManager, requestContext.getUserId(), CREATE_ACCOUNT);
    }

    @Nonnull
    @Override
    public CreateUserAccountResult execute(@Nonnull CreateUserAccountAction action, @Nonnull ExecutionContext executionContext) {
        authenticationManager.registerUser(action.getUserId(), action.getEmailAddress(), action.getPasswordDigest(), action.getSalt());
        return new CreateUserAccountResult();
    }
}
