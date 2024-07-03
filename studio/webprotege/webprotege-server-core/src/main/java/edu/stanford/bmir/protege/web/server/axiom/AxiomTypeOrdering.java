package edu.stanford.bmir.protege.web.server.axiom;



import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 25/02/15
 */

@Target({ElementType.PARAMETER}) @Retention(RetentionPolicy.RUNTIME)
public @interface AxiomTypeOrdering {
}
