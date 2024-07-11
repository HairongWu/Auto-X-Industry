package edu.stanford.bmir.protege.web.server.auth;

import javax.inject.Qualifier;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 17/02/15
 */
@Qualifier
@Retention(RetentionPolicy.RUNTIME)
public @interface ChapSessionMaxDuration {
}
