package edu.stanford.bmir.protege.web.server.mail;

import javax.annotation.Nonnull;
import java.util.List;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 07/02/15
 */
public interface SendMail {

    /**
     * Sends an email to the specified recipients.  The email will have the specified subject and specified content.
     *
     * @param messageId The id of the message to be sent.
     * @param recipientEmailAddresses The email addresses of the recipients.  Not {@code null}.  The email will be
     *                                sent with a content-type header of text/html, which means that html
     *                                can be used to format messages.
     * @param subject The subject of the email.  Not {@code null}.
     * @param text The content of the email.  Not {@code null}.
     * @throws NullPointerException if any parameters are {@code null}.
     */
    void sendMail(@Nonnull MessageId messageId,
                  @Nonnull List<String> recipientEmailAddresses,
                  @Nonnull String subject,
                  @Nonnull String text,
                  @Nonnull MessageHeader... messageHeaders);

    /**
     * Sends an email to the specified recipients.  The email will have the specified subject and specified content.
     *
     * @param recipientEmailAddresses The email addresses of the recipients.  Not {@code null}.  The email will be
     *                                sent with a content-type header of text/html, which means that html
     *                                can be used to format messages.
     * @param subject The subject of the email.  Not {@code null}.
     * @param text The content of the email.  Not {@code null}.
     * @throws NullPointerException if any parameters are {@code null}.
     */
    void sendMail(@Nonnull List<String> recipientEmailAddresses,
                  @Nonnull String subject,
                  @Nonnull String text,
                  @Nonnull MessageHeader... messageHeaders);

    /**
     * Sends an email to the specified recipients.  The email will have the specified subject and specified content.The email will be
     *                                sent with a content-type header of text/html, which means that html
     *                                can be used to format messages.
     *
     * @param recipientEmailAddresses The email addresses of the recipients.  Not {@code null}.
     * @param subject The subject of the email.  Not {@code null}.
     * @param text The content of the email.  Not {@code null}.
     * @param exceptionHandler An exception handler for handling {@link javax.mail.MessagingException}s.  Not {@code null}.
     * @throws NullPointerException if any parameters are {@code null}.
     */
    void sendMail(@Nonnull List<String> recipientEmailAddresses,
                  @Nonnull String subject,
                  @Nonnull String text,
                  @Nonnull MessagingExceptionHandler exceptionHandler,
                  @Nonnull MessageHeader... messageHeaders);

    /**
     * Sends an email to the specified recipients.  The email will have the specified subject and specified content.The email will be
     *                                sent with a content-type header of text/html, which means that html
     *                                can be used to format messages.
     *
     * @param messageId The id of the message to be sent.
     * @param recipientEmailAddresses The email addresses of the recipients.  Not {@code null}.
     * @param subject The subject of the email.  Not {@code null}.
     * @param text The content of the email.  Not {@code null}.
     * @param exceptionHandler An exception handler for handling {@link javax.mail.MessagingException}s.  Not {@code null}.
     * @throws NullPointerException if any parameters are {@code null}.
     */
    void sendMail(@Nonnull MessageId messageId,
                  @Nonnull List<String> recipientEmailAddresses,
                  @Nonnull String subject,
                  @Nonnull String text,
                  @Nonnull MessagingExceptionHandler exceptionHandler,
                  @Nonnull MessageHeader... messageHeaders);
}
